"""
mech_init_ipm.py — 无 FEM 锚定的应力/位移初始化（反幂法思想）

设计依据 2026-09-11 审查定稿的《神经网络反幂法应力位移初始化方案》。
与 mech_init.py 的关系：复用其材料场 / 采样 / 力学量 / 边界损失 / 存图等基础
函数（import，不改动 mech_init.py 任何代码）。目标：去除 L_norm（质量归一化）
与 L_ray / L_pde 中的 ω²_FEM 锚定，ω² 改由 Rayleigh 商读出（输出而非输入）；
FEM 只在预拟合提供初始特征函数时用一次。

本文件实现两种方法（cfg.method 切换）：

(1) method="selfconsistent"（默认，推荐）——自洽特征残差：
        r = ∇·σ^NN + ρ·u^NN / √m̄,   m̄ = 固定评估集上的质量积分（detach）
    不动点：−∇·σ(u) = (1/√m̄)·ρu 与特征方程 −∇·σ(u) = ω²ρu 一致 ⟺ m̄ = 1/ω⁴。
    相当于把原 mech_init 锚定残差 ∇·σ^NN + ω²_FEM·ρ·u^NN 中的锚点 ω²_FEM
    换成网络自身的 1/√m̄（自洽特征值）：u^NN 直接出现在残差中（与原 formulation
    同构，优化性态经过验证），振幅由方程自身钉在 m = 1/ω⁴（u→0 时 ρu/√m̄ 尺度
    不变、残差不消失，故无需 L_norm），ω² = 1/√m̄ 与 Rayleigh 商互为印证。
    流程：FEM 预拟合（选定基频模态盆地）→ Adam 固定步数 → L-BFGS（train 早停）
    → 归一化到 m=1 保存。

(2) method="ipm_outer"（论文 IPMNN 的严格外层反幂形式，保留供对照）：
        外层固定 n_outer 次：冻结 Ũ_{k-1}，内层解 −∇·σ(U_k) = ρ·Ũ_{k-1}
        （Adam 固定步 + L-BFGS），再 Ũ_k = U_k/√m。
    ★ 2026-09-11 实测诊断：该形式在本混合架构下收敛到虚假不动点
    （R 停在 0.606 而非 0.225，overlap→0.999）。原因：u^NN 不直接出现在
    反幂残差中（残差只含冻结的 Ũ），u 的振幅只能经 L_cons（σ 的零阶匹配）
    间接获得；L-BFGS 在固定点集上（8000 点 vs 6.4 万参数）过拟合出
    "σ 通道单独满足平衡方程、u 通道滞后"的解（train 0.19 / val 68）。
    自洽残差（方法 1）让 u^NN 直接进残差，正是针对此问题的修复。

损失（两种方法均为 5 项加权和，掩码统一 S，权重沿用 mech_init 现值）：
    (1) L_eig   特征残差（内部点；w_pde=5 复用为该权重）
    (2) L_cons  本构残差 σ^NN ↔ σ^u（不变）
    (3) L_blk   重块邻域加密点上重算 (1)+(2)
    (4) L_dir   左右固支 u=0（不变）
    (5) L_neu   上下自由 σ_xy=σ_yy=0（不变）

退出条件：预拟合 Adam 固定 prefit_steps 步；selfconsistent 主训练 Adam 固定
adam_steps 步 + L-BFGS train 逐块相对下降连续 early_stop_patience 块
< early_stop_rtol（默认 0.02）早停；ipm_outer 外层固定 n_outer 次、内层同上。

护栏（ipm_outer）：每次外层打印 m、R、与上一 Ũ 的质量内积重叠（应 →1，
<0.9 提示模态切换风险）；m < 1e-12 视为数值坍缩，中止。

父类中 omega2_fem / w_norm / w_ray / rayleigh_warmup / adam_early_switch 等
字段在本文件中不使用（ω²_FEM 仅作诊断对照，从 npz 读取，不进任何损失）。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from networks import NetworkConfig, build_mechanics_networks, MECH_NET_NAMES
from mech_init import (
    MechInitConfig,
    _resolve,
    load_frozen_phi,
    heaviside_s,
    material_fields,
    sample_interior_pool,
    sample_block_pool,
    sample_boundary_pools,
    minibatch,
    mech_quantities,
    constitutive_stress,
    dirichlet_loss,
    neumann_loss,
    save_field_figure,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 配置（继承 MechInitConfig，只改默认值与新增字段）
# ---------------------------------------------------------------------------

@dataclass
class IpmConfig(MechInitConfig):
    # 方法选择："selfconsistent"（默认，自洽特征残差）或 "ipm_outer"（外层反幂，供对照）
    method: str = "selfconsistent"

    # FEM 模态数据（阶段 1 预拟合的唯一 FEM 输入；omega2_fine[0] 仅作诊断对照）
    fem_npz: str = "fem_results/fem_result.npz"

    # 阶段 1 预拟合（监督回归 FEM 第 1 阶模态，选定基频模态盆地）
    prefit_steps: int = 3000
    w_fit: float = 1.0

    # selfconsistent 主训练：Adam 固定步数沿用原初始化预算（父类默认 20000）

    # ipm_outer：外层固定次数 + 每次外层的 Adam 固定步数
    n_outer: int = 10
    outer_adam_steps: int = 2000

    # L-BFGS：train 损失逐块早停（两种方法共用）
    lbfgs_blocks: int = 160
    lbfgs_early_stop: bool = True
    early_stop_metric: str = "train"
    early_stop_rtol: float = 0.02
    early_stop_patience: int = 5

    # 自洽因子 / 外层归一化 / Rayleigh 商的固定评估点集规模（Sobol，只评估不训练）
    norm_eval_points: int = 20000

    # 输出（与既有 *_init.pt 权重和 mech_figures/ 区分开，互不覆盖）
    save_suffix: str = "_ipm.pt"
    fig_dir: str = "mech_figures_ipm"


# ---------------------------------------------------------------------------
# FEM 第 1 阶模态加载（双线性插值；阶段 1 的监督目标）
# ---------------------------------------------------------------------------

class FemMode:
    """fem_result.npz 中第 1 阶模态的双线性插值器。

    npz 约定（fem_solver.save_results）：mode1_ux/mode1_uy 形状 (ny+1, nx+1)，
    [iy, ix] 对应物理坐标 (ix*hx, iy*hy)；模态已质量归一化（φᵀMφ=1）、
    符号约定为最大绝对值分量为正。
    """

    def __init__(self, npz_path: str, dtype: torch.dtype):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"未找到 FEM 模态文件 {npz_path}；请先运行 fem_solver.py 生成（阶段 0）。")
        d = np.load(npz_path)
        self.omega2 = float(d["omega2_fine"][0])
        self.nx, self.ny = int(d["nx"]), int(d["ny"])
        self.lx, self.ly = float(d["lx"]), float(d["ly"])
        self.hx, self.hy = self.lx / self.nx, self.ly / self.ny
        self.ux = torch.as_tensor(np.asarray(d["mode1_ux"]), dtype=dtype)
        self.uy = torch.as_tensor(np.asarray(d["mode1_uy"]), dtype=dtype)
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """xy (N,2) 物理坐标 -> (u_x, u_y) 各 (N,1)，双线性插值。"""
        fx = (xy[:, 0] / self.hx).clamp(0.0, self.nx - 1e-12)
        fy = (xy[:, 1] / self.hy).clamp(0.0, self.ny - 1e-12)
        i0 = fx.floor().long()
        j0 = fy.floor().long()
        tx = (fx - i0.to(self.dtype)).unsqueeze(1)
        ty = (fy - j0.to(self.dtype)).unsqueeze(1)

        def bil(g: torch.Tensor) -> torch.Tensor:
            g00, g10 = g[j0, i0].unsqueeze(1), g[j0, i0 + 1].unsqueeze(1)
            g01, g11 = g[j0 + 1, i0].unsqueeze(1), g[j0 + 1, i0 + 1].unsqueeze(1)
            return ((1 - tx) * (1 - ty) * g00 + tx * (1 - ty) * g10
                    + (1 - tx) * ty * g01 + tx * ty * g11)

        return bil(self.ux), bil(self.uy)


def _scale_output_(net: torch.nn.Module, c: float) -> None:
    """把 NormalizedMLP 的线性输出层缩放 c 倍（输出函数随之精确缩放 c 倍）。"""
    lin = net.net[-1]
    with torch.no_grad():
        lin.weight.mul_(c)
        lin.bias.mul_(c)


# ---------------------------------------------------------------------------
# 阶段 1：预拟合（监督回归 FEM 第 1 阶模态，选定基频模态盆地）
# ---------------------------------------------------------------------------

def prefit_fem(mech, phi_net, fem: FemMode, pools, cfg: IpmConfig,
               gen: torch.Generator, history: list):
    """Adam 固定步数：w_fit·L_fit + w_cons·L_cons + w_dir·L_dir + w_neu·L_neu。

    L_fit 为内部点上 u^NN 与 FEM 模态（双线性插值）的 MSE；L_cons 让 σ 网络
    同步跟上 σ^u，给后续训练热启动；边界项保持边界条件不被拟合带偏。
    """
    if cfg.prefit_steps <= 0:
        print("[Prefit] prefit_steps=0，跳过预拟合（随机初始，不推荐）")
        return
    params = list(mech.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr_init)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.prefit_steps, eta_min=cfg.lr_final)

    print(f"[Prefit] 开始：{cfg.prefit_steps} 步，目标 = FEM 第 1 阶模态（{cfg.fem_npz}，"
          f"ω²_FEM = {fem.omega2:.8f} 仅对照）")
    t0 = time.time()
    for step in range(cfg.prefit_steps):
        xy_i = minibatch(pools["interior"], cfg.batch_interior, gen)
        xy_d = minibatch(pools["dir"], 2 * cfg.batch_boundary_per_edge, gen)
        xy_n = minibatch(pools["neu"], 2 * cfg.batch_boundary_per_edge, gen)

        s, rho_hat, rho, e = material_fields(phi_net, xy_i, cfg)
        q = mech_quantities(mech, xy_i, create_graph=True)
        sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)
        with torch.no_grad():
            t_x, t_y = fem(xy_i)

        l_fit = ((q.u_x - t_x) ** 2 + (q.u_y - t_y) ** 2).mean()
        l_cons = (s * ((q.sxx - sxx_u) ** 2 + (q.syy - syy_u) ** 2
                       + (q.sxy - sxy_u) ** 2)).mean()
        l_dir = dirichlet_loss(mech, phi_net, xy_d, cfg)
        l_neu = neumann_loss(mech, phi_net, xy_n, cfg)
        total = (cfg.w_fit * l_fit + cfg.w_cons * l_cons
                 + cfg.w_dir * l_dir + cfg.w_neu * l_neu)

        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.prefit_steps - 1:
            history.append({"step": step, "total": float(total.detach()),
                            "fit": float(l_fit.detach()),
                            "cons": float(l_cons.detach()),
                            "dir": float(l_dir.detach()),
                            "neu": float(l_neu.detach())})
            print(f"[Prefit] step {step:5d} | total {float(total):.4e} | "
                  f"fit {float(l_fit):.3e} cons {float(l_cons):.3e} | "
                  f"dir {float(l_dir):.3e} neu {float(l_neu):.3e} | "
                  f"{time.time() - t0:.0f}s")
    print(f"[Prefit] 结束，用时 {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# 方法 1：自洽特征残差（selfconsistent）
# ---------------------------------------------------------------------------

def eval_mass(mech, srho_eval: torch.Tensor, eval_pts: torch.Tensor,
              cfg: IpmConfig) -> float:
    """在固定评估点集上计算质量积分 m = |Ω|·mean(S·ρ̂·|u|²)（detach 标量）。

    srho_eval 为预算的 S·ρ̂（评估点集固定，该乘积为常数，主流程中只算一次）。
    """
    with torch.no_grad():
        u_sq = mech["u_x"](eval_pts) ** 2 + mech["u_y"](eval_pts) ** 2
        return float(cfg.area * (srho_eval * u_sq).mean())


def sc_interior_losses(mech, phi_net, xy, inv_sqrt_m: float, cfg: IpmConfig,
                       create_graph: bool = True):
    """自洽特征残差（内部点）：

        r = ∇·σ^NN + ρ·u^NN·(1/√m̄)

    m̄ 为固定评估集上的质量积分（detach 常数，每步重算）。不动点处
    1/√m̄ = ω²（即 m̄ = 1/ω⁴），与原锚定残差同构但锚点自洽、无需 FEM。
    u^NN 直接出现在残差中（振幅直接被 PDE 钉住），这是与 ipm_outer 的关键区别。
    """
    s, rho_hat, rho, e = material_fields(phi_net, xy, cfg)
    q = mech_quantities(mech, xy, create_graph=create_graph)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    r_x = q.div_x + rho * q.u_x * inv_sqrt_m
    r_y = q.div_y + rho * q.u_y * inv_sqrt_m
    l_eig = (s * (r_x ** 2 + r_y ** 2)).mean()

    l_cons = (s * ((q.sxx - sxx_u) ** 2 + (q.syy - syy_u) ** 2
                   + (q.sxy - sxy_u) ** 2)).mean()

    u_sq = q.u_x ** 2 + q.u_y ** 2
    m = cfg.area * (s * rho_hat * u_sq).mean()
    energy_density = sxx_u * q.eps_xx + syy_u * q.eps_yy + sxy_u * q.gamma_xy
    r_ray = (s * energy_density).sum() / ((s * rho_hat * u_sq).sum() + 1e-30)

    return SimpleNamespace(l_eig=l_eig, l_cons=l_cons, m=m, r_ray=r_ray)


def sc_block_losses(mech, phi_net, xy, inv_sqrt_m: float, cfg: IpmConfig,
                    create_graph: bool = True):
    """(3) 重块加密项：重块邻域点上重算自洽特征残差 + 本构残差。"""
    s, rho_hat, rho, e = material_fields(phi_net, xy, cfg)
    q = mech_quantities(mech, xy, create_graph=create_graph)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    r_x = q.div_x + rho * q.u_x * inv_sqrt_m
    r_y = q.div_y + rho * q.u_y * inv_sqrt_m
    l_eig_blk = (s * (r_x ** 2 + r_y ** 2)).mean()
    l_cons_blk = (s * ((q.sxx - sxx_u) ** 2 + (q.syy - syy_u) ** 2
                       + (q.sxy - sxy_u) ** 2)).mean()
    return l_eig_blk, l_cons_blk


def sc_total_loss(mech, phi_net, eval_pts, srho_eval, pts, cfg: IpmConfig,
                  create_graph: bool = True):
    """自洽方法总损失（5 项加权和）。每步先在固定评估集上重算 m̄（detach）。"""
    m_bar = eval_mass(mech, srho_eval, eval_pts, cfg)
    inv_sqrt_m = 1.0 / math.sqrt(m_bar + 1e-30)

    il = sc_interior_losses(mech, phi_net, pts["interior"], inv_sqrt_m, cfg, create_graph)
    b_eig, b_cons = sc_block_losses(mech, phi_net, pts["block"], inv_sqrt_m, cfg, create_graph)
    l_dir = dirichlet_loss(mech, phi_net, pts["dir"], cfg)
    l_neu = neumann_loss(mech, phi_net, pts["neu"], cfg)

    total = (cfg.w_pde * il.l_eig
             + cfg.w_cons * il.l_cons
             + cfg.w_blk * (b_eig + b_cons)
             + cfg.w_dir * l_dir
             + cfg.w_neu * l_neu)

    parts = {
        "total": float(total.detach()),
        "pde": float(il.l_eig.detach()),
        "cons": float(il.l_cons.detach()),
        "blk": float((b_eig + b_cons).detach()),
        "dir": float(l_dir.detach()),
        "neu": float(l_neu.detach()),
        "m": float(il.m.detach()),
        "m_bar": m_bar,
        "R": float(il.r_ray.detach()),
    }
    return total, parts


# ---------------------------------------------------------------------------
# 方法 2：外层反幂（ipm_outer，论文 IPMNN 的严格形式，供对照）
# ---------------------------------------------------------------------------

def ipm_interior_losses(mech, phi_net, u_prev, xy, cfg: IpmConfig,
                        create_graph: bool = True):
    """反幂残差（内部点）：∇·σ^NN + ρ·Ũ_{k-1}（Ũ 冻结、质量归一化，不进梯度）。

    即固定体力 ρŨ 的线性弹性平衡方程残差，对应父类 L_pde 中
    omega2_fem * rho * u^NN 换为 rho * u_prev。
    """
    s, rho_hat, rho, e = material_fields(phi_net, xy, cfg)
    q = mech_quantities(mech, xy, create_graph=create_graph)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    with torch.no_grad():
        up_x = u_prev["u_x"](xy)
        up_y = u_prev["u_y"](xy)

    r_x = q.div_x + rho * up_x
    r_y = q.div_y + rho * up_y
    l_eig = (s * (r_x ** 2 + r_y ** 2)).mean()

    l_cons = (s * ((q.sxx - sxx_u) ** 2 + (q.syy - syy_u) ** 2
                   + (q.sxy - sxy_u) ** 2)).mean()

    u_sq = q.u_x ** 2 + q.u_y ** 2
    m = cfg.area * (s * rho_hat * u_sq).mean()
    energy_density = sxx_u * q.eps_xx + syy_u * q.eps_yy + sxy_u * q.gamma_xy
    r_ray = (s * energy_density).sum() / ((s * rho_hat * u_sq).sum() + 1e-30)

    return SimpleNamespace(l_eig=l_eig, l_cons=l_cons, m=m, r_ray=r_ray)


def ipm_block_losses(mech, phi_net, u_prev, xy, cfg: IpmConfig,
                     create_graph: bool = True):
    """(3) 重块加密项：重块邻域点上重算反幂残差 + 本构残差。"""
    s, rho_hat, rho, e = material_fields(phi_net, xy, cfg)
    q = mech_quantities(mech, xy, create_graph=create_graph)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    with torch.no_grad():
        up_x = u_prev["u_x"](xy)
        up_y = u_prev["u_y"](xy)

    r_x = q.div_x + rho * up_x
    r_y = q.div_y + rho * up_y
    l_eig_blk = (s * (r_x ** 2 + r_y ** 2)).mean()
    l_cons_blk = (s * ((q.sxx - sxx_u) ** 2 + (q.syy - syy_u) ** 2
                       + (q.sxy - sxy_u) ** 2)).mean()
    return l_eig_blk, l_cons_blk


def ipm_total_loss(mech, phi_net, u_prev, pts, cfg: IpmConfig,
                   create_graph: bool = True):
    """外层反幂总损失（5 项加权和）。pts 为 dict：interior / block / dir / neu。"""
    il = ipm_interior_losses(mech, phi_net, u_prev, pts["interior"], cfg, create_graph)
    b_eig, b_cons = ipm_block_losses(mech, phi_net, u_prev, pts["block"], cfg, create_graph)
    l_dir = dirichlet_loss(mech, phi_net, pts["dir"], cfg)
    l_neu = neumann_loss(mech, phi_net, pts["neu"], cfg)

    total = (cfg.w_pde * il.l_eig
             + cfg.w_cons * il.l_cons
             + cfg.w_blk * (b_eig + b_cons)
             + cfg.w_dir * l_dir
             + cfg.w_neu * l_neu)

    parts = {
        "total": float(total.detach()),
        "pde": float(il.l_eig.detach()),
        "cons": float(il.l_cons.detach()),
        "blk": float((b_eig + b_cons).detach()),
        "dir": float(l_dir.detach()),
        "neu": float(l_neu.detach()),
        "m": float(il.m.detach()),
        "R": float(il.r_ray.detach()),
    }
    return total, parts


def normalize_and_update(mech, phi_net, u_prev, eval_pts, cfg: IpmConfig,
                         has_prev: bool):
    """外层归一化：Ũ_k ← U_k/√m(U_k)（迭代步骤，不进训练图）。返回 (m, R, overlap)。

    m、R 在固定评估点集上计算（create_graph=False，只取值）；
    overlap 为新 Ũ 与旧 Ũ 的质量内积（旧 Ũ 有 m=1）：应 →1，<0.9 提示模态切换风险。
    归一化通过把冻结副本输出层缩放 1/√m 实现（线性层缩放 = 输出函数精确缩放）。
    """
    s, rho_hat, rho, e = material_fields(phi_net, eval_pts, cfg)
    q = mech_quantities(mech, eval_pts, create_graph=False)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    u_sq = q.u_x ** 2 + q.u_y ** 2
    m = cfg.area * (s * rho_hat * u_sq).mean()
    energy_density = sxx_u * q.eps_xx + syy_u * q.eps_yy + sxy_u * q.gamma_xy
    r_ray = (s * energy_density).sum() / ((s * rho_hat * u_sq).sum() + 1e-30)

    m_f = float(m)
    if m_f < 1e-12:
        raise RuntimeError(f"[护栏] m = {m_f:.3e} 过小，数值坍缩，中止。"
                           f"（理论上不会发生：反幂残差含固定非零源）")

    with torch.no_grad():
        if has_prev:
            old_x = u_prev["u_x"](eval_pts)
            old_y = u_prev["u_y"](eval_pts)
            overlap = float(cfg.area
                            * (s * rho_hat * (old_x * q.u_x.detach()
                                              + old_y * q.u_y.detach())).mean()
                            / math.sqrt(m_f))
        else:
            overlap = float("nan")

    scale = 1.0 / math.sqrt(m_f)
    for name in ("u_x", "u_y"):
        u_prev[name].load_state_dict(mech[name].state_dict())
        _scale_output_(u_prev[name], scale)
    return m_f, float(r_ray), overlap


# ---------------------------------------------------------------------------
# 通用训练驱动（两种方法共用，损失以闭包注入）
# ---------------------------------------------------------------------------

def train_adam_generic(mech, pools, cfg: IpmConfig, gen: torch.Generator,
                       history: list, loss_fn: Callable, tag: str,
                       adam_steps: int):
    """Adam 固定步数（余弦退火 lr_init→lr_final），minibatch 每步重采。

    loss_fn(pts, create_graph=True) -> (total, parts)。
    """
    if adam_steps <= 0:
        print(f"[{tag}] adam_steps=0，跳过 Adam 段")
        return
    params = list(mech.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr_init)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=adam_steps, eta_min=cfg.lr_final)

    print(f"[{tag}] 开始：{adam_steps} 步（固定），lr {cfg.lr_init}→{cfg.lr_final}（余弦）")
    t0 = time.time()
    for step in range(adam_steps):
        pts = {
            "interior": minibatch(pools["interior"], cfg.batch_interior, gen),
            "block": minibatch(pools["block"], cfg.batch_block, gen),
            "dir": minibatch(pools["dir"], 2 * cfg.batch_boundary_per_edge, gen),
            "neu": minibatch(pools["neu"], 2 * cfg.batch_boundary_per_edge, gen),
        }
        total, parts = loss_fn(pts, True)

        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == adam_steps - 1:
            parts.update({"step": step, "lr": sched.get_last_lr()[0]})
            history.append(parts)
            print(f"[{tag}] step {step:5d} | total {parts['total']:.4e} | "
                  f"pde {parts['pde']:.3e} cons {parts['cons']:.3e} blk {parts['blk']:.3e} | "
                  f"dir {parts['dir']:.3e} neu {parts['neu']:.3e} | "
                  f"m {parts['m']:.4f} R {parts['R']:.4f} | "
                  f"lr {parts['lr']:.2e} | {time.time() - t0:.0f}s")
    print(f"[{tag}] 结束，用时 {time.time() - t0:.0f}s")


def train_lbfgs_generic(mech, train_pts, val_pts, cfg: IpmConfig,
                        history: list, loss_fn: Callable, tag: str):
    """L-BFGS 精调（固定点集，强 Wolfe），train 损失逐块早停：
    相对下降连续 early_stop_patience 块 < early_stop_rtol 即停。"""
    params = list(mech.parameters())
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
        total, _ = loss_fn(train_pts, True)
        total.backward()
        return total

    prev_metric: Optional[float] = None
    stall = 0
    print(f"[{tag}] 开始：≤{cfg.lbfgs_blocks} 块 × {cfg.lbfgs_max_iter} 次，强 Wolfe；"
          f"早停 {cfg.early_stop_metric} 相对下降 < {cfg.early_stop_rtol} "
          f"连续 {cfg.early_stop_patience} 块")
    t0 = time.time()
    for block in range(cfg.lbfgs_blocks):
        opt.step(closure)

        train_loss = float(loss_fn(train_pts, False)[0].detach())
        val_loss = float(loss_fn(val_pts, False)[0].detach())

        metric = train_loss if cfg.early_stop_metric == "train" else val_loss
        drop_rel = (float("inf") if prev_metric is None
                    else (prev_metric - metric) / max(abs(prev_metric), 1e-30))
        prev_metric = metric
        stall = stall + 1 if drop_rel < cfg.early_stop_rtol else 0

        history.append({"block": block, "train": train_loss, "val": val_loss})
        print(f"[{tag}] block {block:3d} | train {train_loss:.6e} | "
              f"val {val_loss:.6e} | drop {drop_rel:+.2e} | stall {stall} | "
              f"{time.time() - t0:.0f}s")

        if cfg.lbfgs_early_stop and stall >= cfg.early_stop_patience:
            print(f"[{tag}] 早停：连续 {stall} 块 {cfg.early_stop_metric} "
                  f"损失相对下降 < {cfg.early_stop_rtol}")
            break
    print(f"[{tag}] 结束，用时 {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# 诊断（自洽读出：残差用本次读出的 R，不含 ω²_FEM）
# ---------------------------------------------------------------------------

def diagnose_ipm(mech, phi_net, cfg: IpmConfig, omega2_ref: float):
    """训练后自检：Rayleigh 商（读出）vs FEM 参考（仅对照）、m、各项残差 RMS。"""
    gen = torch.Generator().manual_seed(cfg.seed + 300)
    pts = sample_interior_pool(cfg.val_interior, cfg, cfg.seed + 301)
    s, rho_hat, rho, e = material_fields(phi_net, pts, cfg)
    q = mech_quantities(mech, pts, create_graph=False)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    u_sq = q.u_x ** 2 + q.u_y ** 2
    m = cfg.area * (s * rho_hat * u_sq).mean()
    energy_density = sxx_u * q.eps_xx + syy_u * q.eps_yy + sxy_u * q.gamma_xy
    r_ray = (s * energy_density).sum() / ((s * rho_hat * u_sq).sum() + 1e-30)

    # 自洽特征方程残差：∇·σ^NN + R·ρ·u^NN（R 为本次读出值，非 FEM 锚点）
    r_x = q.div_x + r_ray * rho * q.u_x
    r_y = q.div_y + r_ray * rho * q.u_y
    res_rms = torch.sqrt((s * (r_x ** 2 + r_y ** 2)).mean())

    dir_pts, neu_pts = sample_boundary_pools(cfg.val_boundary_per_edge, cfg, gen)
    s_d = heaviside_s(phi_net, dir_pts, cfg)
    dir_rms = torch.sqrt((s_d * (mech["u_x"](dir_pts) ** 2
                                 + mech["u_y"](dir_pts) ** 2)).mean())
    s_n = heaviside_s(phi_net, neu_pts, cfg)
    neu_rms = torch.sqrt((s_n * (mech["sigma_xy"](neu_pts) ** 2
                                 + mech["sigma_yy"](neu_pts) ** 2)).mean())

    rel_err = abs(float(r_ray) - omega2_ref) / omega2_ref
    print("\n===== 诊断（无锚定，自洽读出） =====")
    print(f"Rayleigh 商 R（= ω² 读出）   = {float(r_ray):.6f}")
    print(f"FEM 参考 ω²_FEM（仅对照）    = {omega2_ref:.6f}")
    print(f"相对误差 |R−ω²|/ω²           = {rel_err:.3e}")
    print(f"质量积分 m（归一化后应 ≈1）  = {float(m):.6f}")
    print(f"自洽 PDE 残差 RMS（S 加权）  = {float(res_rms):.3e}")
    print(f"Dirichlet 残差 RMS           = {float(dir_rms):.3e}")
    print(f"Neumann 残差 RMS             = {float(neu_rms):.3e}")
    return {"R": float(r_ray), "m": float(m), "rel_err": rel_err,
            "res_rms": float(res_rms)}


# ---------------------------------------------------------------------------
# 存图（场图复用 mech_init.save_field_figure；收敛曲线两种方法各一版）
# ---------------------------------------------------------------------------

def save_convergence_figure_sc(adam_history: list, lbfgs_history: list,
                               omega2_ref: float, cfg: IpmConfig, path: str):
    """selfconsistent：Adam 总损失 / L-BFGS train+val / R 随 Adam 步收敛曲线。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.4), constrained_layout=True)

    if adam_history:
        xs = [h["step"] for h in adam_history]
        ax1.semilogy(xs, [h["total"] for h in adam_history], lw=0.8, label="total")
        ax1.semilogy(xs, [h["pde"] for h in adam_history], lw=0.8, label="pde(eig)")
        ax1.semilogy(xs, [h["cons"] for h in adam_history], lw=0.8, label="cons")
        ax1.semilogy(xs, [h["blk"] for h in adam_history], lw=0.8, label="blk")
        ax1.legend(fontsize=8)
    ax1.set_xlabel("Adam step")
    ax1.set_ylabel("loss")
    ax1.set_title("Stage A (Adam, self-consistent)")
    ax1.grid(alpha=0.3)

    if lbfgs_history:
        xs = list(range(len(lbfgs_history)))
        ax2.semilogy(xs, [h["train"] for h in lbfgs_history], "o-", ms=3, label="train")
        ax2.semilogy(xs, [h["val"] for h in lbfgs_history], "s-", ms=3, label="val")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("L-BFGS block")
    ax2.set_ylabel("total loss")
    ax2.set_title("Stage B (L-BFGS)")
    ax2.grid(alpha=0.3)

    if adam_history:
        xs = [h["step"] for h in adam_history]
        ax3.plot(xs, [h["R"] for h in adam_history], lw=0.8, label="PINN Rayleigh R")
        ax3.axhline(omega2_ref, color="C3", ls="--",
                    label=f"FEM omega2 = {omega2_ref:.6f} (ref only)")
        ax3.legend(fontsize=8)
    ax3.set_xlabel("Adam step")
    ax3.set_ylabel("R")
    ax3.set_title("Rayleigh quotient convergence")
    ax3.grid(alpha=0.3)

    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_convergence_figure_outer(adam_history: list, lbfgs_history: list,
                                  r_history: list, omega2_ref: float,
                                  cfg: IpmConfig, path: str):
    """ipm_outer：Adam 总损失（外层拼接）/ L-BFGS train+val / R 随外层收敛曲线。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.4), constrained_layout=True)

    if adam_history:
        xs = list(range(len(adam_history)))
        ax1.semilogy(xs, [h["total"] for h in adam_history], lw=0.8)
        bounds, cur = [], 0
        for k in range(1, cfg.n_outer + 1):
            n_k = sum(1 for h in adam_history if h.get("outer") == k)
            cur += n_k
            bounds.append(cur)
        for b in bounds[:-1]:
            ax1.axvline(b, color="gray", ls="--", lw=0.6)
    ax1.set_xlabel("logged Adam steps (outer concatenated, dashed = outer boundary)")
    ax1.set_ylabel("total loss")
    ax1.set_title("IPM inner Adam")
    ax1.grid(alpha=0.3)

    if lbfgs_history:
        xs = list(range(len(lbfgs_history)))
        ax2.semilogy(xs, [h["train"] for h in lbfgs_history], "o-", ms=3, label="train")
        ax2.semilogy(xs, [h["val"] for h in lbfgs_history], "s-", ms=3, label="val")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("logged L-BFGS blocks (outer concatenated)")
    ax2.set_ylabel("total loss")
    ax2.set_title("IPM inner L-BFGS")
    ax2.grid(alpha=0.3)

    ks = list(range(len(r_history)))
    ax3.plot(ks, r_history, "o-", label="PINN Rayleigh R")
    ax3.axhline(omega2_ref, color="C3", ls="--",
                label=f"FEM omega2 = {omega2_ref:.6f} (ref only)")
    ax3.set_xlabel("outer iteration (0 = FEM prefit)")
    ax3.set_ylabel("R")
    ax3.set_title("Rayleigh quotient convergence")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(cfg: IpmConfig | None = None):
    cfg = cfg or IpmConfig()
    cfg.weights_dir = _resolve(cfg.weights_dir)
    cfg.fig_dir = _resolve(cfg.fig_dir)
    cfg.phi_path = _resolve(cfg.phi_path)
    cfg.fem_npz = _resolve(cfg.fem_npz)
    os.makedirs(cfg.weights_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    # 冻结的 SDF → 材料场
    phi_net = load_frozen_phi(cfg)
    print(f"已加载冻结 SDF：{cfg.phi_path}")
    print(f"方法：{cfg.method}")

    # 5 个力学网络（与 mech_init 同结构）
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly,
                            hidden_layers=cfg.hidden_layers,
                            hidden_width=cfg.hidden_width,
                            dtype=cfg.dtype)
    mech = build_mechanics_networks(net_cfg)
    n_params = sum(p.numel() for p in mech.parameters())
    print(f"力学网络：{list(mech.keys())}，总参数 {n_params}")

    # 阶段 0 产物：FEM 第 1 阶模态（唯一使用 FEM 之处）
    fem = FemMode(cfg.fem_npz, cfg.dtype)
    print(f"FEM 模态：{cfg.fem_npz}（{fem.nx}×{fem.ny} 网格，"
          f"ω²_FEM = {fem.omega2:.8f} 仅作对照）")

    # Adam 点池（与 mech_init 同口径）
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    pools = {
        "interior": sample_interior_pool(cfg.pool_interior, cfg, cfg.seed + 2),
        "block": sample_block_pool(cfg.pool_block, cfg, gen),
        "dir": None, "neu": None,
    }
    pools["dir"], pools["neu"] = sample_boundary_pools(
        cfg.pool_boundary_per_edge, cfg, gen)

    # L-BFGS 固定训练点集 + 独立验证集（只评估不训练，不同种子）
    gen_train = torch.Generator().manual_seed(cfg.seed + 100)
    gen_val = torch.Generator().manual_seed(cfg.seed + 200)
    train_pts = {
        "interior": sample_interior_pool(cfg.lbfgs_interior, cfg, cfg.seed + 101),
        "block": sample_block_pool(cfg.lbfgs_block, cfg, gen_train),
        "dir": None, "neu": None,
    }
    train_pts["dir"], train_pts["neu"] = sample_boundary_pools(
        cfg.lbfgs_boundary_per_edge, cfg, gen_train)
    val_pts = {
        "interior": sample_interior_pool(cfg.val_interior, cfg, cfg.seed + 201),
        "block": sample_block_pool(cfg.val_block, cfg, gen_val),
        "dir": None, "neu": None,
    }
    val_pts["dir"], val_pts["neu"] = sample_boundary_pools(
        cfg.val_boundary_per_edge, cfg, gen_val)

    # 固定评估点集（自洽因子 m̄ / 外层归一化 / Rayleigh 读出共用），预算 S·ρ̂
    eval_pts = sample_interior_pool(cfg.norm_eval_points, cfg, cfg.seed + 400)
    with torch.no_grad():
        s_ev, rho_hat_ev, _, _ = material_fields(phi_net, eval_pts, cfg)
        srho_eval = s_ev * rho_hat_ev

    # 阶段 1：预拟合 FEM 模态（选定基频模态盆地）
    prefit_history: list = []
    prefit_fem(mech, phi_net, fem, pools, cfg, gen, prefit_history)

    adam_history: list = []
    lbfgs_history: list = []
    r_history: list = []

    if cfg.method == "selfconsistent":
        # 阶段 2：Adam 固定步数 + L-BFGS（train 早停），单阶段自洽训练
        def loss_fn(pts, create_graph):
            return sc_total_loss(mech, phi_net, eval_pts, srho_eval, pts, cfg,
                                 create_graph)

        m0 = eval_mass(mech, srho_eval, eval_pts, cfg)
        print(f"[SC] 预拟合后：m = {m0:.4f}（收敛目标 m → 1/ω⁴ ≈ "
              f"{1.0 / fem.omega2 ** 2:.1f}）")
        train_adam_generic(mech, pools, cfg, gen, adam_history, loss_fn,
                           "SC-Adam", cfg.adam_steps)
        train_lbfgs_generic(mech, train_pts, val_pts, cfg, lbfgs_history,
                            loss_fn, "SC-LBFGS")
        m_last = eval_mass(mech, srho_eval, eval_pts, cfg)
        r_last = float("nan")  # 由 diagnose 统一读出
    elif cfg.method == "ipm_outer":
        # 阶段 2：固定 n_outer 次外层反幂迭代（内层 = Adam 固定步 + L-BFGS）
        _up = build_mechanics_networks(net_cfg)
        u_prev = torch.nn.ModuleDict({"u_x": _up["u_x"], "u_y": _up["u_y"]})
        u_prev.requires_grad_(False)

        def loss_fn(pts, create_graph):
            return ipm_total_loss(mech, phi_net, u_prev, pts, cfg, create_graph)

        m_last, r0, _ = normalize_and_update(mech, phi_net, u_prev, eval_pts, cfg,
                                             has_prev=False)
        print(f"[IPM] Ũ_0（FEM 预拟合）：m = {m_last:.4f}，R = {r0:.6f}"
              f"（FEM 参考 {fem.omega2:.6f}）")
        r_history.append(r0)
        for k in range(1, cfg.n_outer + 1):
            print(f"\n===== 外层反幂迭代 {k}/{cfg.n_outer} =====")
            train_adam_generic(mech, pools, cfg, gen, adam_history, loss_fn,
                               f"IPM-Adam][outer {k}", cfg.outer_adam_steps)
            train_lbfgs_generic(mech, train_pts, val_pts, cfg, lbfgs_history,
                                loss_fn, f"IPM-LBFGS][outer {k}")
            m_last, r_last, overlap = normalize_and_update(
                mech, phi_net, u_prev, eval_pts, cfg, has_prev=True)
            r_history.append(r_last)
            warn = "  [警告] 重叠 < 0.9，模态切换风险！" if overlap < 0.9 else ""
            print(f"[IPM] outer {k}：m = {m_last:.4f}，R = {r_last:.6f}，"
                  f"overlap = {overlap:.4f}{warn}")
            if not cfg.quiet_figures:
                save_field_figure(mech, phi_net, cfg,
                                  os.path.join(cfg.fig_dir, f"outer{k:02d}.png"),
                                  title=f"IPM outer {k}, R={r_last:.5f}")
    else:
        raise ValueError(f"未知 method：{cfg.method}")

    # 阶段 3：把训练态网络归一化到 m=1（u、σ 同步缩放；线性层缩放精确），保存
    final_scale = 1.0 / math.sqrt(m_last)
    for name in MECH_NET_NAMES:
        _scale_output_(mech[name], final_scale)
    print(f"\n收尾归一化：全部力学网络输出层 × 1/√m = {final_scale:.4f}（m=1 约定）")

    saved = []
    for name in MECH_NET_NAMES:
        p = os.path.join(cfg.weights_dir, f"{name}{cfg.save_suffix}")
        torch.save(mech[name].state_dict(), p)
        saved.append(p)
    print("已保存力学网络权重（m=1 归一化）：")
    for p in saved:
        print(f"  {p}")

    if cfg.method == "selfconsistent":
        save_convergence_figure_sc(adam_history, lbfgs_history, fem.omega2,
                                   cfg, os.path.join(cfg.fig_dir, "convergence.png"))
    else:
        save_convergence_figure_outer(adam_history, lbfgs_history, r_history,
                                      fem.omega2, cfg,
                                      os.path.join(cfg.fig_dir, "convergence.png"))
    save_field_figure(mech, phi_net, cfg,
                      os.path.join(cfg.fig_dir, "final.png"),
                      title="final (no FEM anchor)")

    diag = diagnose_ipm(mech, phi_net, cfg, fem.omega2)

    result = {
        "method": cfg.method,
        "omega2_rayleigh": diag["R"],
        "omega2_fem_ref": fem.omega2,
        "rel_err_vs_fem": diag["rel_err"],
        "r_history": r_history,
        "prefit_steps": cfg.prefit_steps,
        "adam_steps": cfg.adam_steps if cfg.method == "selfconsistent"
        else cfg.outer_adam_steps,
        "n_outer": cfg.n_outer if cfg.method == "ipm_outer" else None,
        "diag": diag,
    }
    json_path = os.path.join(cfg.weights_dir, "mech_ipm_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {json_path}")
    print("无锚定力学初始化完成。")
    return diag


if __name__ == "__main__":
    main()
