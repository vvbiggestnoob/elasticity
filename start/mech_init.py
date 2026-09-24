"""
mech_init.py — 应力/位移（力学）初始化

对应《应力位移初始化说明.md》：在初始（全材料）几何上训练 5 个力学网络
(u_x, u_y, σ_xx, σ_xy, σ_yy)，使其满足弹性特征值方程

    −∇·σ = ω²ρu

FEM 只提供特征值标量锚点 ω²_FEM（Rayleigh 商锚定 + 事后对拍），不做逐点监督。

材料场由冻结的 SDF 网络给出（加载 weights/phi_init.pt）：
    S = ½(1+tanh(φ/2β))，  ρ = S·ρ̂，  E = E_void + (E_solid − E_void)·S  （ersatz 插值）

损失为 7 项加权和（掩码统一为 S）：
    (1) L_pde   平衡方程残差（内部点）
    (2) L_cons  本构残差（σ 网络绑定到 u 网络的应变）
    (3) L_blk   重块邻域加密点上再算 (1)+(2)
    (4) L_dir   左右固支边界 u=0
    (5) L_neu   上下自由边界 σ_xy=σ_yy=0
    (6) L_norm  质量归一化 (m−1)²，排除 u≡0 平凡解
    (7) L_ray   Rayleigh 商锚定（Adam 前 warmup 步关闭）

两阶段训练：
    阶段 A：Adam，Sobol 低差异点池 + 每步 minibatch 重采，余弦退火 lr 1e-3→1e-5；
    阶段 B：L-BFGS 固定点集全批量，分块执行，独立验证集早停，回滚到验证最优权重。

数值全部取自《实验设置与计算范围.md》。全程 float64。
"""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import plot_utils as pu

pu.setup_style()  # 中文标签字体 + 统一黑白风格

from networks import NetworkConfig, build_mechanics_networks, build_sdf_network, MECH_NET_NAMES

# 所有相对路径相对本脚本所在目录解析，从任意工作目录运行均可
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class MechInitConfig:
    # 几何（《实验设置与计算范围.md》§3）
    lx: float = 1.6
    ly: float = 0.5

    # 力学网络结构：4 层 × 64 tanh（每个 12737 参数；SDF 网络 phi 保持 3 层不变，
    # 其权重已训练好且结构与之绑定）。改层数后旧权重不兼容，需重新训练。
    hidden_layers: int = 4
    hidden_width: int = 64

    # 材料参数（§4）
    e_solid: float = 1.0
    e_void: float = 1e-6
    nu: float = 0.3
    rho_base: float = 1.0
    rho_block: float = 100.0

    # 重块区域（§5，不可优化）
    block_x: tuple = (0.76, 0.84)
    block_y: tuple = (0.21, 0.29)
    # 重块加密点的采样邻域：重块矩形向外扩 block_margin（实现自定义，见使用文档）
    block_margin: float = 0.06

    # SDF / Heaviside（§7）
    beta: float = 0.01

    # FEM 特征值锚点（§13，全材料+重块）。2026-09-03 由 fem_solver.py 实测回填：
    # 160×50 细网格 ω²_FEM = 0.22517705（粗网格 0.19397494，Richardson 0.23557776）。
    # 注意：既有力学网络权重是按旧占位锚点 0.238 训练的，以新锚点为准需重跑本脚本。
    omega2_fem: float = 0.22517705

    # 损失权重（《应力位移初始化说明.md》§4 参考权重）
    w_pde: float = 5.0
    w_cons: float = 1.0
    w_blk: float = 0.1
    w_dir: float = 10.0
    w_neu: float = 1.0
    w_norm: float = 10.0
    w_ray: float = 5.0

    # Adam 采样规模（§11）：点池 + 每步 minibatch
    pool_interior: int = 20000
    pool_block: int = 3000
    pool_boundary_per_edge: int = 2000
    batch_interior: int = 4096
    batch_block: int = 512
    batch_boundary_per_edge: int = 256   # 每条边 256；Dirichlet/Neumann 各 2 条边

    # L-BFGS 固定点集（§11）
    lbfgs_interior: int = 8000
    lbfgs_block: int = 1000
    lbfgs_boundary_per_edge: int = 300

    # 独立验证集（只评估不训练，不同种子）
    val_interior: int = 8192
    val_block: int = 1024
    val_boundary_per_edge: int = 256

    # 训练超参数（§12）
    adam_steps: int = 20000
    lr_init: float = 1e-3
    lr_final: float = 1e-5
    rayleigh_warmup: int = 0           # 前 500 步关闭 L_ray
    # Adam 停滞提前切换：过 warmup 且满 min_step 后，window 步内 total 相对改善
    # < rtol 则提前进入 L-BFGS（Adam 尾部 lr≈1e-5 推进很慢，让给精调更快）
    adam_early_switch: bool = False
    adam_switch_min_step: int = 2000
    adam_switch_window: int = 300
    adam_switch_rtol: float = 5e-3
    lbfgs_blocks: int = 160
    lbfgs_max_iter: int = 70
    lbfgs_history: int = 50
    # 验证集早停 / 回滚：默认关闭——配点即"网格"，目标是充分拟合固定点集；
    # val 列仍然打印供观察，但不干预训练。需要时置 True 恢复说明文档 §6 的行为。
    lbfgs_early_stop: bool = False
    lbfgs_rollback: bool = False
    early_stop_patience: int = 5         # 连续若干块观察损失相对下降不足则早停
    early_stop_rtol: float = 1e-4        # 逐块相对下降阈值（交替循环中力学取 0.1）
    # 早停观察的损失："val"（验证集，防过拟合固定点集）或 "train"（训练损失，
    # 拟合到收敛才停，更贴合"配点即网格"的定位）；回滚始终按 val
    early_stop_metric: str = "val"

    # 日志 / 存图
    log_every: int = 100                 # Adam 每多少步打印
    fig_every: int = 500                 # Adam 每多少步存图
    fig_every_blocks: int = 5            # L-BFGS 每多少块存图（末块必存）
    fig_nx: int = 161                    # 诊断/存图网格（与 FEM 细网格节点一致）
    fig_ny: int = 51

    # 路径与随机种子
    weights_dir: str = "weights"
    fig_dir: str = "mech_figures"
    phi_path: str = "weights/phi_init.pt"
    save_suffix: str = "_init.pt"
    seed: int = 20240901

    # 交替循环用（默认值保持单独运行行为不变）：
    # phi_t 为 SDF 求值的伪时间切片（初始化阶段恒 0；循环第 k 步几何在 t=k·dt）；
    # init_suffix 非 None 时从 weights_dir/*<init_suffix> 热启动力学网络；
    # quiet_figures=True 时跳过 Adam/L-BFGS 过程图与初始图，只存最终场图与损失图
    phi_t: float = 0.0
    init_suffix: Optional[str] = None
    quiet_figures: bool = False
    # 出图标签（plot_utils.fig_path）：非空时按类型分文件夹保存
    # （{fig_dir}/{kind}/{fig_tag}.png，交替循环传 iter{k:03d}）；空 = 平铺文件名
    fig_tag: str = ""

    dtype: torch.dtype = torch.float64

    @property
    def area(self) -> float:
        return self.lx * self.ly


# ---------------------------------------------------------------------------
# 材料场（冻结 SDF → S, ρ̂, ρ, E）
# ---------------------------------------------------------------------------

def load_frozen_phi(cfg: MechInitConfig) -> torch.nn.Module:
    """加载 SDF 初始化得到的 phi 网络并冻结（初始化阶段 t 恒取 0）。"""
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype)
    phi = build_sdf_network(net_cfg)
    phi.load_state_dict(torch.load(_resolve(cfg.phi_path), weights_only=True))
    phi.eval()
    phi.requires_grad_(False)
    return phi


def heaviside_s(phi_net: torch.nn.Module, xy: torch.Tensor, cfg: MechInitConfig) -> torch.Tensor:
    """材料场 S = ½(1+tanh(φ/2β))。SDF 冻结，S 作为常数权重，不需要梯度。

    SDF 在 t = cfg.phi_t 切片上取值：初始化阶段 phi_t=0；交替循环第 k 步
    几何为 phi_iter{k} 在 t=k·dt 的切片。
    """
    with torch.no_grad():
        tt = torch.full((xy.shape[0], 1), cfg.phi_t, dtype=cfg.dtype)
        phi_val = phi_net(torch.cat([xy, tt], dim=1))
        s = 0.5 * (1.0 + torch.tanh(phi_val / (2.0 * cfg.beta)))
    return s


def block_indicator(xy: torch.Tensor, cfg: MechInitConfig) -> torch.Tensor:
    """重块指示函数（硬边界）：重块矩形内为 1，其余为 0，返回 (N,1)。"""
    x, y = xy[:, 0:1], xy[:, 1:2]
    inside = (
        (x >= cfg.block_x[0]) & (x <= cfg.block_x[1])
        & (y >= cfg.block_y[0]) & (y <= cfg.block_y[1])
    )
    return inside.to(cfg.dtype)


def material_fields(phi_net: torch.nn.Module, xy: torch.Tensor, cfg: MechInitConfig):
    """返回 (S, rho_hat, rho, E)，均为 (N,1) 常数张量（相对力学网络不可导）。"""
    s = heaviside_s(phi_net, xy, cfg)
    rho_hat = cfg.rho_base + (cfg.rho_block - cfg.rho_base) * block_indicator(xy, cfg)
    rho = s * rho_hat
    e = cfg.e_void + (cfg.e_solid - cfg.e_void) * s   # ersatz 插值（设计决策，见文档）
    return s, rho_hat, rho, e


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------

def sample_interior_pool(n: int, cfg: MechInitConfig, seed: int) -> torch.Tensor:
    """Sobol 低差异内部点池，(n,2) 物理坐标。"""
    eng = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    pts = eng.draw(n).to(cfg.dtype)
    return pts * torch.tensor([cfg.lx, cfg.ly], dtype=cfg.dtype)


def sample_block_pool(n: int, cfg: MechInitConfig, gen: torch.Generator) -> torch.Tensor:
    """重块邻域加密点池：重块矩形外扩 block_margin 的盒内均匀采样（裁剪到域内）。"""
    x0 = max(0.0, cfg.block_x[0] - cfg.block_margin)
    x1 = min(cfg.lx, cfg.block_x[1] + cfg.block_margin)
    y0 = max(0.0, cfg.block_y[0] - cfg.block_margin)
    y1 = min(cfg.ly, cfg.block_y[1] + cfg.block_margin)
    u = torch.rand(n, 2, generator=gen, dtype=cfg.dtype)
    lo = torch.tensor([x0, y0], dtype=cfg.dtype)
    hi = torch.tensor([x1, y1], dtype=cfg.dtype)
    return lo + u * (hi - lo)


def sample_boundary_pools(n_per_edge: int, cfg: MechInitConfig, gen: torch.Generator):
    """四条边界各自的点池。返回 (dir_pool, neu_pool)，各 (2*n_per_edge, 2)。

    Dirichlet：左 x=0、右 x=Lx（固支 u=0）；
    Neumann：下 y=0、上 y=Ly（自由 σ_xy=σ_yy=0）。
    """
    t1 = torch.rand(n_per_edge, 1, generator=gen, dtype=cfg.dtype)
    t2 = torch.rand(n_per_edge, 1, generator=gen, dtype=cfg.dtype)
    t3 = torch.rand(n_per_edge, 1, generator=gen, dtype=cfg.dtype)
    t4 = torch.rand(n_per_edge, 1, generator=gen, dtype=cfg.dtype)

    left = torch.cat([torch.zeros_like(t1), t1 * cfg.ly], dim=1)
    right = torch.cat([torch.full_like(t2, cfg.lx), t2 * cfg.ly], dim=1)
    bottom = torch.cat([t3 * cfg.lx, torch.zeros_like(t3)], dim=1)
    top = torch.cat([t4 * cfg.lx, torch.full_like(t4, cfg.ly)], dim=1)

    dir_pool = torch.cat([left, right], dim=0)
    neu_pool = torch.cat([bottom, top], dim=0)
    return dir_pool, neu_pool


def minibatch(pool: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """从点池随机抽 minibatch（有放回）。"""
    idx = torch.randint(0, pool.shape[0], (n,), generator=gen)
    return pool[idx]


# ---------------------------------------------------------------------------
# 力学量：前向 + 一阶自动微分（物理坐标导数，归一化在网络内部）
# ---------------------------------------------------------------------------

def mech_quantities(mech: torch.nn.ModuleDict, xy: torch.Tensor, create_graph: bool = True):
    """在 xy 上计算全部力学量（一阶自动微分）。

    返回命名空间：
        u_x, u_y, sxx, sxy, syy            网络输出 (N,1)
        eps_xx, eps_yy, gamma_xy           应变（u 网络的一阶导）
        div_x, div_y                       ∇·σ 的两个分量（σ 网络的一阶导）
        sxx_u, syy_u, sxy_u                由应变经平面应力本构算出的"目标"应力

    create_graph=True 供训练（损失对网络参数可导）；False 供验证/诊断（只取值）。
    """
    xy = xy.detach().requires_grad_(True)

    u_x = mech["u_x"](xy)
    u_y = mech["u_y"](xy)
    sxx = mech["sigma_xx"](xy)
    sxy = mech["sigma_xy"](xy)
    syy = mech["sigma_yy"](xy)

    du_x = torch.autograd.grad(u_x.sum(), xy, create_graph=create_graph)[0]
    du_y = torch.autograd.grad(u_y.sum(), xy, create_graph=create_graph)[0]
    dsxx = torch.autograd.grad(sxx.sum(), xy, create_graph=create_graph)[0]
    dsxy = torch.autograd.grad(sxy.sum(), xy, create_graph=create_graph)[0]
    dsyy = torch.autograd.grad(syy.sum(), xy, create_graph=create_graph)[0]

    eps_xx = du_x[:, 0:1]
    eps_yy = du_y[:, 1:2]
    gamma_xy = du_x[:, 1:2] + du_y[:, 0:1]

    div_x = dsxx[:, 0:1] + dsxy[:, 1:2]
    div_y = dsxy[:, 0:1] + dsyy[:, 1:2]

    return SimpleNamespace(
        u_x=u_x, u_y=u_y, sxx=sxx, sxy=sxy, syy=syy,
        eps_xx=eps_xx, eps_yy=eps_yy, gamma_xy=gamma_xy,
        div_x=div_x, div_y=div_y,
    )


def constitutive_stress(q, e: torch.Tensor, cfg: MechInitConfig):
    """平面应力本构：由应变算目标应力 σ^u。e 为 (N,1) 弹性模量场。"""
    nu = cfg.nu
    factor = e / (1.0 - nu * nu)
    sxx_u = factor * (q.eps_xx + nu * q.eps_yy)
    syy_u = factor * (q.eps_yy + nu * q.eps_xx)
    sxy_u = e / (2.0 * (1.0 + nu)) * q.gamma_xy
    return sxx_u, syy_u, sxy_u


# ---------------------------------------------------------------------------
# 损失（7 项加权和，掩码统一 S）
# ---------------------------------------------------------------------------

def interior_losses(mech, phi_net, xy, cfg: MechInitConfig, create_graph: bool = True):
    """内部点上的 (1)(2)(6)(7) 相关量。返回命名空间（张量，标量）。"""
    s, rho_hat, rho, e = material_fields(phi_net, xy, cfg)
    q = mech_quantities(mech, xy, create_graph=create_graph)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    # (1) 平衡方程残差：∇·σ^NN + ω²_FEM·ρ·u^NN
    r_x = q.div_x + cfg.omega2_fem * rho * q.u_x
    r_y = q.div_y + cfg.omega2_fem * rho * q.u_y
    l_pde = (s * (r_x ** 2 + r_y ** 2)).mean()

    # (2) 本构残差：σ^NN ↔ σ^u
    l_cons = (s * ((q.sxx - sxx_u) ** 2 + (q.syy - syy_u) ** 2 + (q.sxy - sxy_u) ** 2)).mean()

    # (6) 质量归一化：m ≈ ∫ S ρ̂ |u|² dΩ（蒙特卡洛，乘域面积）
    u_sq = q.u_x ** 2 + q.u_y ** 2
    m = cfg.area * (s * rho_hat * u_sq).mean()

    # (7) Rayleigh 商：R = Σ S σ^u:ε / Σ S ρ̂ |u|²（振幅不变）
    energy_density = sxx_u * q.eps_xx + syy_u * q.eps_yy + sxy_u * q.gamma_xy
    r_ray = (s * energy_density).sum() / ((s * rho_hat * u_sq).sum() + 1e-30)

    return SimpleNamespace(l_pde=l_pde, l_cons=l_cons, m=m, r_ray=r_ray)


def block_losses(mech, phi_net, xy, cfg: MechInitConfig, create_graph: bool = True):
    """(3) 重块加密项：重块邻域点上再算一遍平衡 + 本构残差。"""
    s, rho_hat, rho, e = material_fields(phi_net, xy, cfg)
    q = mech_quantities(mech, xy, create_graph=create_graph)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    r_x = q.div_x + cfg.omega2_fem * rho * q.u_x
    r_y = q.div_y + cfg.omega2_fem * rho * q.u_y
    l_pde_blk = (s * (r_x ** 2 + r_y ** 2)).mean()
    l_cons_blk = (s * ((q.sxx - sxx_u) ** 2 + (q.syy - syy_u) ** 2 + (q.sxy - sxy_u) ** 2)).mean()
    return l_pde_blk, l_cons_blk


def dirichlet_loss(mech, phi_net, xy, cfg: MechInitConfig):
    """(4) 左右固支边界：S·(u_x²+u_y²)。无需导数。"""
    s = heaviside_s(phi_net, xy, cfg)
    u_x = mech["u_x"](xy)
    u_y = mech["u_y"](xy)
    return (s * (u_x ** 2 + u_y ** 2)).mean()


def neumann_loss(mech, phi_net, xy, cfg: MechInitConfig):
    """(5) 上下自由边界：S·(σ_xy²+σ_yy²)。无需导数。"""
    s = heaviside_s(phi_net, xy, cfg)
    sxy = mech["sigma_xy"](xy)
    syy = mech["sigma_yy"](xy)
    return (s * (sxy ** 2 + syy ** 2)).mean()


def compute_total_loss(mech, phi_net, pts, cfg: MechInitConfig,
                       use_ray: bool, create_graph: bool = True):
    """总损失。pts 为 dict：interior / block / dir / neu 四个点集。

    返回 (total, parts)，parts 为各项损失的浮点字典（含 m 与 R，供日志）。
    """
    il = interior_losses(mech, phi_net, pts["interior"], cfg, create_graph)
    l_pde_blk, l_cons_blk = block_losses(mech, phi_net, pts["block"], cfg, create_graph)
    l_dir = dirichlet_loss(mech, phi_net, pts["dir"], cfg)
    l_neu = neumann_loss(mech, phi_net, pts["neu"], cfg)

    l_norm = (il.m - 1.0) ** 2
    l_ray = (il.r_ray / cfg.omega2_fem - 1.0) ** 2

    total = (
        cfg.w_pde * il.l_pde
        + cfg.w_cons * il.l_cons
        + cfg.w_blk * (l_pde_blk + l_cons_blk)
        + cfg.w_dir * l_dir
        + cfg.w_neu * l_neu
        + cfg.w_norm * l_norm
    )
    if use_ray:
        total = total + cfg.w_ray * l_ray

    parts = {
        "total": float(total.detach()),
        "pde": float(il.l_pde.detach()),
        "cons": float(il.l_cons.detach()),
        "blk": float((l_pde_blk + l_cons_blk).detach()),
        "dir": float(l_dir.detach()),
        "neu": float(l_neu.detach()),
        "norm": float(l_norm.detach()),
        "ray": float(l_ray.detach()) if use_ray else float("nan"),
        "m": float(il.m.detach()),
        "R": float(il.r_ray.detach()),
    }
    return total, parts


def evaluate_loss(mech, phi_net, pts, cfg: MechInitConfig):
    """验证/诊断用：只取总损失值（一阶导不建图，省内存）。"""
    total, _ = compute_total_loss(mech, phi_net, pts, cfg, use_ray=True, create_graph=False)
    return float(total.detach())


# ---------------------------------------------------------------------------
# 阶段 A：Adam（点池 + 每步 minibatch 重采，余弦退火，Rayleigh warmup）
# ---------------------------------------------------------------------------

def train_adam(mech, phi_net, pools, cfg: MechInitConfig, gen: torch.Generator, history: list):
    if cfg.adam_steps <= 0:
        print("[Adam] adam_steps=0，跳过 Adam 段，直接进入 L-BFGS（随机初始化起点）")
        return
    params = list(mech.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr_init)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.adam_steps, eta_min=cfg.lr_final
    )

    print(f"[Adam] 开始：{cfg.adam_steps} 步，lr {cfg.lr_init}→{cfg.lr_final}（余弦），"
          f"前 {cfg.rayleigh_warmup} 步关闭 L_ray")
    t0 = time.time()
    for step in range(cfg.adam_steps):
        pts = {
            "interior": minibatch(pools["interior"], cfg.batch_interior, gen),
            "block": minibatch(pools["block"], cfg.batch_block, gen),
            "dir": minibatch(pools["dir"], 2 * cfg.batch_boundary_per_edge, gen),
            "neu": minibatch(pools["neu"], 2 * cfg.batch_boundary_per_edge, gen),
        }
        use_ray = step >= cfg.rayleigh_warmup
        total, parts = compute_total_loss(mech, phi_net, pts, cfg, use_ray=use_ray)

        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.adam_steps - 1:
            parts["step"] = step
            parts["lr"] = sched.get_last_lr()[0]
            history.append(parts)
            print(f"[Adam] step {step:5d} | total {parts['total']:.4e} | "
                  f"pde {parts['pde']:.3e} cons {parts['cons']:.3e} blk {parts['blk']:.3e} | "
                  f"dir {parts['dir']:.3e} neu {parts['neu']:.3e} norm {parts['norm']:.3e} "
                  f"ray {parts['ray']:.3e} | m {parts['m']:.4f} R {parts['R']:.4f} | "
                  f"lr {parts['lr']:.2e} | {time.time()-t0:.0f}s")

            # 停滞检测：Adam 尾部推进缓慢时提前切到 L-BFGS
            w_logs = cfg.adam_switch_window // cfg.log_every
            if (cfg.adam_early_switch and step >= cfg.adam_switch_min_step
                    and len(history) > w_logs):
                old = history[-1 - w_logs]["total"]
                rel = (old - parts["total"]) / max(abs(old), 1e-30)
                if rel < cfg.adam_switch_rtol:
                    print(f"[Adam] 停滞检测：{cfg.adam_switch_window} 步内 total 相对改善 "
                          f"{rel:.2e} < {cfg.adam_switch_rtol}，提前切换到 L-BFGS")
                    break

        if not cfg.quiet_figures and (step % cfg.fig_every == 0
                                      or step == cfg.adam_steps - 1):
            save_field_figures(mech, phi_net, cfg, tag=f"adam{step:06d}")
    print(f"[Adam] 结束，用时 {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# 阶段 B：L-BFGS 精调（固定点集，分块，独立验证集早停 + 回滚）
# ---------------------------------------------------------------------------

def train_lbfgs(mech, phi_net, cfg: MechInitConfig, history: list):
    params = list(mech.parameters())
    gen_train = torch.Generator().manual_seed(cfg.seed + 100)
    gen_val = torch.Generator().manual_seed(cfg.seed + 200)

    # 固定训练点集（保证损失确定性）
    train_pts = {
        "interior": sample_interior_pool(cfg.lbfgs_interior, cfg, cfg.seed + 101),
        "block": sample_block_pool(cfg.lbfgs_block, cfg, gen_train),
        "dir": None, "neu": None,
    }
    train_pts["dir"], train_pts["neu"] = sample_boundary_pools(
        cfg.lbfgs_boundary_per_edge, cfg, gen_train)

    # 独立验证集（不同种子，只评估不训练）
    val_pts = {
        "interior": sample_interior_pool(cfg.val_interior, cfg, cfg.seed + 201),
        "block": sample_block_pool(cfg.val_block, cfg, gen_val),
        "dir": None, "neu": None,
    }
    val_pts["dir"], val_pts["neu"] = sample_boundary_pools(
        cfg.val_boundary_per_edge, cfg, gen_val)

    best_val = math.inf
    best_state = copy.deepcopy(mech.state_dict())
    prev_metric: Optional[float] = None   # 早停观察量（early_stop_metric）的上一块值
    stall = 0

    print(f"[L-BFGS] 开始：≤{cfg.lbfgs_blocks} 块 × {cfg.lbfgs_max_iter} 次，强 Wolfe，"
          f"固定点集 {cfg.lbfgs_interior}+{cfg.lbfgs_block}+2×{cfg.lbfgs_boundary_per_edge}")
    # 优化器全程只建一次：曲率历史（逆 Hessian 近似）跨块累积，收敛更快；
    # 块边界只用于打印 / 观察 val / 存图，不打断优化状态。
    opt = torch.optim.LBFGS(
        params,
        max_iter=cfg.lbfgs_max_iter,
        history_size=cfg.lbfgs_history,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-16,
        tolerance_change=1e-16,
    )

    def closure():
        opt.zero_grad()
        total, _ = compute_total_loss(mech, phi_net, train_pts, cfg,
                                      use_ray=True, create_graph=True)
        total.backward()
        return total

    t0 = time.time()
    for block in range(cfg.lbfgs_blocks):
        opt.step(closure)

        train_loss = evaluate_loss(mech, phi_net, train_pts, cfg)
        val_loss = evaluate_loss(mech, phi_net, val_pts, cfg)

        rel_improve = (best_val - val_loss) / max(abs(best_val), 1e-30)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            if cfg.lbfgs_rollback:
                best_state = copy.deepcopy(mech.state_dict())
        # 早停：early_stop_metric 选择的损失逐块相对下降连续 patience 块
        # < early_stop_rtol 则退出（回滚仍按 val）
        metric = train_loss if cfg.early_stop_metric == "train" else val_loss
        drop_rel = (float("inf") if prev_metric is None
                    else (prev_metric - metric) / max(abs(prev_metric), 1e-30))
        prev_metric = metric
        if drop_rel < cfg.early_stop_rtol:
            stall += 1
        else:
            stall = 0

        history.append({"step": f"lbfgs_{block}", "total": train_loss, "val": val_loss})
        mark = " *best" if is_best else ""
        print(f"[L-BFGS] block {block:3d} | train {train_loss:.6e} | "
              f"val {val_loss:.6e} | rel_improve {rel_improve:+.2e}{mark} | "
              f"{time.time()-t0:.0f}s")

        if not cfg.quiet_figures and (block % cfg.fig_every_blocks == 0
                                      or block == cfg.lbfgs_blocks - 1):
            save_field_figures(mech, phi_net, cfg, tag=f"lbfgs{block:03d}")

        if cfg.lbfgs_early_stop and stall >= cfg.early_stop_patience:
            print(f"[L-BFGS] 早停：连续 {stall} 块 {cfg.early_stop_metric} "
                  f"损失相对下降 < {cfg.early_stop_rtol}")
            break

    if cfg.lbfgs_rollback:
        mech.load_state_dict(best_state)
        print(f"[L-BFGS] 结束，已回滚到验证损失最优的权重（best val = {best_val:.6e}），"
              f"用时 {time.time()-t0:.0f}s")
    else:
        print(f"[L-BFGS] 结束，保留最终权重（不回滚；best val = {best_val:.6e} 仅供参考），"
              f"用时 {time.time()-t0:.0f}s")
    return best_val


# ---------------------------------------------------------------------------
# 诊断（《应力位移初始化说明.md》§7）
# ---------------------------------------------------------------------------

@torch.no_grad()
def _grid_points(cfg: MechInitConfig) -> torch.Tensor:
    xs = torch.linspace(0.0, cfg.lx, cfg.fig_nx, dtype=cfg.dtype)
    ys = torch.linspace(0.0, cfg.ly, cfg.fig_ny, dtype=cfg.dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def diagnose(mech, phi_net, cfg: MechInitConfig):
    """训练后自检：Rayleigh 商对拍、质量积分、PDE 残差 RMS、边界残差 RMS。"""
    gen = torch.Generator().manual_seed(cfg.seed + 300)
    pts = sample_interior_pool(cfg.val_interior, cfg, cfg.seed + 301)
    s, rho_hat, rho, e = material_fields(phi_net, pts, cfg)
    q = mech_quantities(mech, pts, create_graph=False)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    u_sq = q.u_x ** 2 + q.u_y ** 2
    m = cfg.area * (s * rho_hat * u_sq).mean()
    energy_density = sxx_u * q.eps_xx + syy_u * q.eps_yy + sxy_u * q.gamma_xy
    r_ray = (s * energy_density).sum() / ((s * rho_hat * u_sq).sum() + 1e-30)

    r_x = q.div_x + cfg.omega2_fem * rho * q.u_x
    r_y = q.div_y + cfg.omega2_fem * rho * q.u_y
    res_rms = torch.sqrt((s * (r_x ** 2 + r_y ** 2)).mean())

    dir_pts, neu_pts = sample_boundary_pools(cfg.val_boundary_per_edge, cfg, gen)
    s_d = heaviside_s(phi_net, dir_pts, cfg)
    dir_rms = torch.sqrt((s_d * (mech["u_x"](dir_pts) ** 2
                                 + mech["u_y"](dir_pts) ** 2)).mean())
    s_n = heaviside_s(phi_net, neu_pts, cfg)
    neu_rms = torch.sqrt((s_n * (mech["sigma_xy"](neu_pts) ** 2
                                 + mech["sigma_yy"](neu_pts) ** 2)).mean())

    rel_err = abs(float(r_ray) - cfg.omega2_fem) / cfg.omega2_fem
    print("\n===== 诊断（训练后自检） =====")
    print(f"Rayleigh 商 R          = {float(r_ray):.6f}")
    print(f"FEM 锚点 ω²_FEM        = {cfg.omega2_fem:.6f}")
    print(f"相对误差 |R−ω²|/ω²     = {rel_err:.3e}")
    print(f"质量积分 m（目标 1）   = {float(m):.6f}")
    print(f"PDE 残差 RMS（S 加权） = {float(res_rms):.3e}")
    print(f"Dirichlet 残差 RMS     = {float(dir_rms):.3e}")
    print(f"Neumann 残差 RMS       = {float(neu_rms):.3e}")
    print("（与 FEM 模态的 MAC / 加权 L2 需 FEM 模态数据，当前阶段暂无，留待对拍。）")
    return {"R": float(r_ray), "m": float(m), "rel_err": rel_err,
            "res_rms": float(res_rms)}


# ---------------------------------------------------------------------------
# 存图（统一风格见 plot_utils：一图一文件、黑白为主、硬块红框、四角不标记）
# ---------------------------------------------------------------------------

def _fig_path(cfg: MechInitConfig, kind: str, tag: Optional[str] = None) -> str:
    """kind = 类型（子文件夹名）；tag 默认取 cfg.fig_tag（循环传 iter{k:03d}）。"""
    return pu.fig_path(cfg.fig_dir, kind, cfg.fig_tag if tag is None else tag)


def _grid_XYn(cfg: MechInitConfig):
    """存图网格的 numpy 坐标矩阵 (ny, nx)。"""
    import numpy as np
    xs = np.linspace(0.0, cfg.lx, cfg.fig_nx)
    ys = np.linspace(0.0, cfg.ly, cfg.fig_ny)
    return np.meshgrid(xs, ys)  # indexing="xy" 默认：(ny, nx)


@torch.no_grad()
def save_field_figures(mech, phi_net, cfg: MechInitConfig,
                       tag: Optional[str] = None):
    """位移/应力场乘材料场 S（孔洞区域置 0），六个场各一张单图：
    u_mag / u_x / u_y / sigma_xx / sigma_xy / sigma_yy。

    S 在当前几何切片（t = cfg.phi_t）上取值；网格上逐点取值，无导数。
    |u| 用 gray_r（黑 = 大），有符号场用 gray 对称色标（黑 = 负、白 = 正）。
    """
    pts = _grid_points(cfg)
    s = heaviside_s(phi_net, pts, cfg).reshape(cfg.fig_nx, cfg.fig_ny)
    u_x = mech["u_x"](pts).reshape(cfg.fig_nx, cfg.fig_ny) * s
    u_y = mech["u_y"](pts).reshape(cfg.fig_nx, cfg.fig_ny) * s
    sxx = mech["sigma_xx"](pts).reshape(cfg.fig_nx, cfg.fig_ny) * s
    sxy = mech["sigma_xy"](pts).reshape(cfg.fig_nx, cfg.fig_ny) * s
    syy = mech["sigma_yy"](pts).reshape(cfg.fig_nx, cfg.fig_ny) * s
    u_mag = torch.sqrt(u_x ** 2 + u_y ** 2)

    fields = [
        ("u_mag", u_mag, "|u|·S", pu.CMAP_SEQ, False),
        ("u_x", u_x, "u_x·S", pu.CMAP_DIV, True),
        ("u_y", u_y, "u_y·S", pu.CMAP_DIV, True),
        ("sigma_xx", sxx, "σ_xx·S", pu.CMAP_DIV, True),
        ("sigma_xy", sxy, "σ_xy·S", pu.CMAP_DIV, True),
        ("sigma_yy", syy, "σ_yy·S", pu.CMAP_DIV, True),
    ]
    X, Y = _grid_XYn(cfg)
    block = (cfg.block_x, cfg.block_y)
    for kind, f, name, cmap, sym in fields:
        pu.save_field(_fig_path(cfg, kind, tag), X, Y,
                      f.T.numpy(),   # (nx,ny) -> (ny,nx)
                      lx=cfg.lx, ly=cfg.ly, cmap=cmap, symmetric=sym,
                      block=block, title=f"{name}（t = {cfg.phi_t}）")


def save_loss_figures(adam_history: list, lbfgs_history: list, cfg: MechInitConfig):
    """损失曲线两张单图：loss_mech_adam（Adam 各项分量）、loss_mech_lbfgs
    （L-BFGS 训练/验证总损失）。"""
    if adam_history:
        steps = [h["step"] for h in adam_history]
        series = []
        for key, label in [("total", "total"), ("pde", "pde"), ("cons", "cons"),
                           ("blk", "blk"), ("dir", "dir"), ("neu", "neu"),
                           ("norm", "norm"), ("ray", "ray")]:
            vals = [h[key] for h in adam_history]
            if any(math.isnan(v) for v in vals):
                # warmup 期间 ray 为 nan，分段画
                xs = [s for s, v in zip(steps, vals) if not math.isnan(v)]
                ys = [v for v in vals if not math.isnan(v)]
                if xs:
                    series.append((xs, ys, label))
            else:
                series.append((steps, vals, label))
        pu.save_lines(_fig_path(cfg, "loss_mech_adam"), series, logy=True,
                      title=f"Mech Stage A (Adam)，ray warmup = "
                            f"{cfg.rayleigh_warmup} 步",
                      xlabel="Adam step", ylabel="loss")
    if lbfgs_history:
        blocks = list(range(len(lbfgs_history)))
        pu.save_lines(_fig_path(cfg, "loss_mech_lbfgs"),
                      [(blocks, [h["total"] for h in lbfgs_history], "train"),
                       (blocks, [h["val"] for h in lbfgs_history], "val")],
                      logy=True, title="Mech Stage B (L-BFGS)",
                      xlabel="L-BFGS block", ylabel="total loss")


# --- 兼容包装（mech_init_ipm.py / mech_init_v2.py 等旧脚本仍在用旧签名） ---

def save_field_figure(mech, phi_net, cfg: MechInitConfig, path: str, title: str = ""):
    """旧接口兼容：path 的目录作图根、文件名词干作 tag，改写六张单图。"""
    d = os.path.dirname(path)
    tag = os.path.splitext(os.path.basename(path))[0]
    old_dir, old_tag = cfg.fig_dir, cfg.fig_tag
    try:
        cfg.fig_dir, cfg.fig_tag = d, tag
        save_field_figures(mech, phi_net, cfg)
    finally:
        cfg.fig_dir, cfg.fig_tag = old_dir, old_tag


def save_loss_figure(adam_history: list, lbfgs_history: list, cfg: MechInitConfig, path: str):
    """旧接口兼容：改写 loss_mech_adam.png / loss_mech_lbfgs.png 到 path 的目录。"""
    d = os.path.dirname(path)
    old_dir, old_tag = cfg.fig_dir, cfg.fig_tag
    try:
        cfg.fig_dir, cfg.fig_tag = d, ""
        save_loss_figures(adam_history, lbfgs_history, cfg)
    finally:
        cfg.fig_dir, cfg.fig_tag = old_dir, old_tag


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(cfg: MechInitConfig | None = None):
    cfg = cfg or MechInitConfig()
    cfg.weights_dir = _resolve(cfg.weights_dir)
    cfg.fig_dir = _resolve(cfg.fig_dir)
    cfg.phi_path = _resolve(cfg.phi_path)
    os.makedirs(cfg.weights_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    # 冻结的 SDF → 材料场
    phi_net = load_frozen_phi(cfg)
    print(f"已加载冻结 SDF：{cfg.phi_path}")

    # 5 个力学网络（hidden_layers×64 tanh，内置归一化，float64）
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly,
                            hidden_layers=cfg.hidden_layers,
                            hidden_width=cfg.hidden_width,
                            dtype=cfg.dtype)
    mech = build_mechanics_networks(net_cfg)
    n_params = sum(p.numel() for p in mech.parameters())
    print(f"力学网络：{list(mech.keys())}，总参数 {n_params}")

    # 交替循环：init_suffix 非 None 时从上一步力学权重热启动（几何小步演化，
    # 力学解变化小，热启动可大幅减少重解步数）
    if cfg.init_suffix is not None:
        for name in MECH_NET_NAMES:
            p = os.path.join(cfg.weights_dir, f"{name}{cfg.init_suffix}")
            mech[name].load_state_dict(torch.load(p, weights_only=True))
        print(f"力学网络热启动：{cfg.weights_dir}/*{cfg.init_suffix}")

    # 初始状态存图
    if not cfg.quiet_figures:
        save_field_figures(mech, phi_net, cfg, tag="initial")

    # Adam 点池（Sobol 内部点 + 重块邻域加密 + 边界点）
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    pools = {
        "interior": sample_interior_pool(cfg.pool_interior, cfg, cfg.seed + 2),
        "block": sample_block_pool(cfg.pool_block, cfg, gen),
        "dir": None, "neu": None,
    }
    pools["dir"], pools["neu"] = sample_boundary_pools(
        cfg.pool_boundary_per_edge, cfg, gen)
    print(f"Adam 点池：interior {cfg.pool_interior}，block {cfg.pool_block}，"
          f"dir/neu 各 2×{cfg.pool_boundary_per_edge}")

    # 阶段 A
    adam_history: list = []
    train_adam(mech, phi_net, pools, cfg, gen, adam_history)

    # 阶段 B
    lbfgs_history: list = []
    train_lbfgs(mech, phi_net, cfg, lbfgs_history)

    # 收尾：损失曲线、最终状态图、保存权重、诊断
    save_loss_figures(adam_history, lbfgs_history, cfg)
    save_field_figures(mech, phi_net, cfg)

    saved = []
    for name in MECH_NET_NAMES:
        p = os.path.join(cfg.weights_dir, f"{name}{cfg.save_suffix}")
        torch.save(mech[name].state_dict(), p)
        saved.append(p)
    print("已保存力学网络权重：")
    for p in saved:
        print(f"  {p}")

    diag = diagnose(mech, phi_net, cfg)
    print("力学初始化完成。")
    return diag


if __name__ == "__main__":
    main()
