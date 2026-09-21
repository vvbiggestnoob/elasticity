# -*- coding: utf-8 -*-
"""
hjb_step.py — HJB 方程驱动的单次 SDF 演化（拓扑优化主循环 Step B）

对应《HJB方程优化说明.md》（唯一权威说明），数值取自《实验设置与计算范围.md》。
当前计算范围（说明文档 §2）：冻结力学网络、V_n 场算一次后固定、α 固定 1.0、
单次演化推进一个 dt、不进入外层交替迭代、不触发重初始化。

流程：
  1. 加载冻结的 5 个力学网络（weights/u_x_init.pt 等）与 SDF 网络（weights/phi_init.pt）；
     SDF 网络加载两份：phi_ref（冻结，提供初始条件锚点与"演化前"状态）和
     phi（可训练，从 phi_ref 热启动）。
  2. 在当前几何上由力学网络计算一次 ω²（Rayleigh 商）、面积 ∫S dx，
     构造法向速度场并在本次演化中固定：
         V_n = ε:A:ε − ω²·ρ·|u|² − (1/α)(∫S dx − C)
     其中 ε:A:ε = σ^u:ε（平面应力本构，ersatz E 插值），ρ = S·ρ̂。
  3. 训练 phi 拟合 t ∈ [t_current, t_current + dt] 上的 HJB 解，损失加权和
     （均不乘掩码 S）：
         L_r       HJB 残差（内部点，(x,y,t) 时空点），λ_r = 1
         L_0       初始条件（t = t_current 切片锚定 phi_ref；含边界带加密点，
                   权重 w_ic_band = 10，锚定 rim），λ_0 = 1
         L_anchor  制造解锚点（可选，λ_anchor 默认 1）：锚定 φ 到一阶制造解
                   phi_ref + (t−t0)·V_n·|∇phi_ref|，破除 HJB 残差的平解退化
                   （常函数残差为零，单靠 L_0 无法把剖面钉在整个时间柱面上），
                   置 lam_anchor = 0 退回纯 PINN
     注：2026-09-08 起移除了外边界零法向梯度损失 L_b——参考论文（DNN Level Set
     TO）对 SDF 不施加域边界条件；且 L_b 与 L_0 在 t=0 棱线上冲突（初始 SDF 在
     边界上 |∂φ/∂n|=1）。移除后边界值由 PDE + 内部信息自然决定（出流式），
     边界可自由后退。
  4. 两阶段训练：Adam（Sobol 点池 + 每步 minibatch 重采，余弦退火 1e-3→1e-5）
     → L-BFGS（固定点集全批量，强 Wolfe，优化器跨块复用，独立验证集只观察，
     早停/回滚默认关闭）。
  5. 诊断（说明文档 §9）+ 出图 + 保存 weights/phi_iter1.pt 与 weights/hjb_state.json。

符号约定（重要，见使用文档 §5.1）：
  材料 = {φ > 0}，∇φ 在边界处指向材料内部。V_n 按说明文档 §5 的公式计算，
  其物理含义为"沿材料外法向的速度"（V_n > 0 边界外扩、加材料）。
  与此一致的水平集方程为 ∂φ/∂t = V_n·|∇φ|，即残差 r = φ_t − V_n·|∇φ|
  （与参考论文 DNN Level Set TO 式 (5) 同号；说明文档 §4 方程的 "+" 号为笔误，
  若按 "+" 实现则边界沿反方向运动、F 上升，与 §5 最速下降推导及 §9"F 应下降"
  的诊断相矛盾）。

全程 float64。所有相对路径相对本脚本所在目录解析。
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 无界面后端，只保存图片
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from torch.quasirandom import SobolEngine

from networks import NetworkConfig, build_mechanics_networks, build_sdf_network, MECH_NET_NAMES

# 所有相对路径相对本脚本所在目录解析，从任意工作目录运行均可
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


# ---------------------------------------------------------------------------
# 配置（数值取自《实验设置与计算范围.md》与《HJB方程优化说明.md》）
# ---------------------------------------------------------------------------

@dataclass
class HJBConfig:
    # 几何（§3）
    lx: float = 1.6
    ly: float = 0.5

    # 材料参数（§4）
    e_solid: float = 1.0
    e_void: float = 1e-6
    nu: float = 0.3
    rho_base: float = 1.0
    rho_block: float = 100.0

    # 重块区域（§5，不可优化，硬指示函数）
    block_x: tuple = (0.76, 0.84)
    block_y: tuple = (0.21, 0.29)

    # SDF / Heaviside（§7）
    beta: float = 0.01

    # 力学网络结构（加载用，必须与 mech_init.py 训练时一致）
    mech_hidden_layers: int = 4
    mech_hidden_width: int = 64

    # 优化目标（《HJB方程优化说明》§1、§2）
    v_target: float = 0.4          # C = 50% × 0.8
    alpha: float = 1.0             # 罚参数，当前阶段固定

    # 演化区间
    t_current: float = 0.0         # 本次演化起点（初始阶段为 0）
    dt: float = 0.1              # 伪时间步长（§7）

    # 损失权重（说明文档 §7 参考权重；L_b 已于 2026-09-08 移除）
    lam_r: float = 1.0
    lam_0: float = 0.1
    # 制造解锚点权重：锚定 φ 到一阶制造解 phi_ref + (t−t0)·V_n·|∇phi_ref|，
    # 破除 HJB 残差的平解退化（置 0 关闭，退回纯 PINN）
    lam_anchor: float = 0.5

    # 阶段 A：Adam 点池（Sobol 低差异）
    pool_interior: int = 20000     # (x,y,t) 3D Sobol
    pool_init: int = 10000         # (x,y) 2D Sobol，t = t_current
    pool_init_band: int = 4000     # 初始条件边界带点（带宽 4β，隔点严格压边，见 §5.3）
    w_ic_band: float = 10.0        # 初始条件边界带权重（同 sdf_init 的 w_bnd）
    # 阶段 A：每步 minibatch（说明文档 §8：N_r=2000, N_0=1000）
    batch_interior: int = 4000
    batch_init: int = 1000
    # 阶段 A：训练超参数（§8：lr 1e-3→1e-5 余弦；步数暂定 100~500，取上限）
    adam_steps: int = 8000
    lr_init: float = 5e-4
    lr_final: float = 5e-5
    # 余弦退火上限：lr 在 anneal_steps 步内从 lr_init 余弦降到 lr_final，
    # 之后保持 lr_final 不变；退火跨度不超过 adam_steps
    anneal_steps: int = 500

    # 阶段 B：L-BFGS 固定点集（实现自定，与力学初始化同量级）
    lbfgs_interior: int = 8192
    lbfgs_init: int = 4096
    lbfgs_init_band: int = 1024
    lbfgs_blocks: int = 100
    lbfgs_max_iter: int = 70
    lbfgs_history: int = 50
    # 早停 / 回滚：默认关闭（充分拟合固定配点集为目标，验证集只打印观察）
    lbfgs_early_stop: bool = False
    lbfgs_rollback: bool = False
    early_stop_patience: int = 5
    early_stop_rtol: float = 1e-4

    # 独立验证集（只评估不训练，不同种子）
    val_interior: int = 4096
    val_init: int = 2048
    val_init_band: int = 512

    # 诊断：积分点集规模（ω² / 面积 / F 的蒙特卡洛积分）、零水平集附近 |∇φ| 统计带
    diag_n: int = 65536
    zero_band: float = 0.05        # |φ| ≤ 0.05 视为零水平集附近
    fem_check: bool = True         # 演化后几何的 FEM ω² 对拍（需 scipy）

    # 日志 / 存图
    log_every: int = 50
    fig_every: int = 100           # Adam 每多少步存图（末步必存）
    fig_every_blocks: int = 5      # L-BFGS 每多少块存图（末块必存）
    fig_nx: int = 321              # 存图网格
    fig_ny: int = 101

    # 路径
    phi_path: str = "weights/phi_init.pt"       # 输入：当前 SDF
    mech_dir: str = "weights"                   # 输入：5 个力学网络
    mech_suffix: str = "_init.pt"
    phi_out: str = "weights/phi_iter1.pt"       # 输出：演化后 SDF（说明文档 §10）
    state_out: str = "weights/hjb_state.json"   # 输出：伪时间与诊断记录
    fig_dir: str = "hjb_figures"
    seed: int = 20260908

    dtype: torch.dtype = torch.float64

    @property
    def area(self) -> float:
        return self.lx * self.ly

    @property
    def t_new(self) -> float:
        return self.t_current + self.dt


# ---------------------------------------------------------------------------
# 点集数据结构（V_n / 初始条件目标均为预计算常数）
# ---------------------------------------------------------------------------

@dataclass
class InteriorSet:
    """HJB 残差点：(x,y) 空间坐标 + t 伪时间 + 预计算的 V_n 与一阶制造解锚点。

    phi_taylor = phi_ref(t0) + (t−t0)·V_n·|∇phi_ref|：HJB 残差只约束比值
    φ_t/|∇φ| = V_n（常函数是残差为零的退化解），单靠初始条件无法把剖面
    钉在整个时间柱面上；制造解锚点给出显式剖面目标，破除退化（见使用文档 §5.3）。
    """
    xy: torch.Tensor         # (N,2)
    t: torch.Tensor          # (N,1)
    vn: torch.Tensor         # (N,1)
    phi_taylor: torch.Tensor # (N,1)，一阶制造解目标值

    def subset(self, idx: torch.Tensor) -> "InteriorSet":
        return InteriorSet(self.xy[idx], self.t[idx], self.vn[idx], self.phi_taylor[idx])


@dataclass
class InitSet:
    """初始条件点：t = t_current 切片上锚定 phi_ref 的输出。

    w 为逐点权重：内部点 1，边界带点 w_ic_band（边界带锚定 rim 处的 φ，
    否则光滑网络会把边界拐角抹圆、rim 处 φ 虚增——见使用文档 §5.3）。
    """
    xy: torch.Tensor      # (N,2)
    t: torch.Tensor       # (N,1)，恒为 t_current
    phi0: torch.Tensor    # (N,1)，phi_ref(xy, t_current)
    w: torch.Tensor       # (N,1)，逐点权重

    def subset(self, idx: torch.Tensor) -> "InitSet":
        return InitSet(self.xy[idx], self.t[idx], self.phi0[idx], self.w[idx])


@dataclass
class PointSets:
    interior: InteriorSet
    init: InitSet


# ---------------------------------------------------------------------------
# 网络加载（力学冻结；SDF 双份：冻结参考 + 可训练热启动）
# ---------------------------------------------------------------------------

def load_frozen_mech(cfg: HJBConfig) -> torch.nn.ModuleDict:
    """加载 5 个力学网络并冻结（本阶段不训练力学网络）。"""
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly,
                            hidden_layers=cfg.mech_hidden_layers,
                            hidden_width=cfg.mech_hidden_width,
                            dtype=cfg.dtype)
    mech = build_mechanics_networks(net_cfg)
    for name in MECH_NET_NAMES:
        path = _resolve(os.path.join(cfg.mech_dir, name + cfg.mech_suffix))
        mech[name].load_state_dict(torch.load(path, weights_only=True))
    mech.eval()
    mech.requires_grad_(False)
    return mech


def load_phi(cfg: HJBConfig, frozen: bool) -> torch.nn.Module:
    """从 cfg.phi_path 加载 SDF 网络。frozen=True 为参考锚点，False 为可训练副本。"""
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype)
    phi = build_sdf_network(net_cfg)
    phi.load_state_dict(torch.load(_resolve(cfg.phi_path), weights_only=True))
    if frozen:
        phi.eval()
        phi.requires_grad_(False)
    return phi


# ---------------------------------------------------------------------------
# 材料场与力学量（冻结网络，一阶自动微分取应变）
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_phi_at(phi: torch.nn.Module, xy: torch.Tensor, t_val: float) -> torch.Tensor:
    """在指定 t 切片上评估 phi。xy: (N,2) -> (N,1)。"""
    t = torch.full((xy.shape[0], 1), t_val, dtype=xy.dtype, device=xy.device)
    return phi(torch.cat([xy, t], dim=1))


def block_indicator(xy: torch.Tensor, cfg: HJBConfig) -> torch.Tensor:
    """重块指示函数（硬边界）：重块矩形内为 1，其余为 0，返回 (N,1)。"""
    x, y = xy[:, 0:1], xy[:, 1:2]
    inside = (
        (x >= cfg.block_x[0]) & (x <= cfg.block_x[1])
        & (y >= cfg.block_y[0]) & (y <= cfg.block_y[1])
    )
    return inside.to(cfg.dtype)


def fields_from_phi_val(phi_val: torch.Tensor, xy: torch.Tensor, cfg: HJBConfig):
    """由 phi 值给出 (S, rho_hat, rho, E)，均为 (N,1) 常数张量。"""
    s = 0.5 * (1.0 + torch.tanh(phi_val / (2.0 * cfg.beta)))
    rho_hat = cfg.rho_base + (cfg.rho_block - cfg.rho_base) * block_indicator(xy, cfg)
    rho = s * rho_hat
    e = cfg.e_void + (cfg.e_solid - cfg.e_void) * s   # ersatz 插值
    return s, rho_hat, rho, e


def frozen_strain(mech: torch.nn.ModuleDict, xy: torch.Tensor):
    """冻结力学网络上的一阶自动微分：返回 detach 的 (u_x, u_y, eps_xx, eps_yy, gamma_xy)。

    只需 u 网络的一阶导（应变）；σ 网络不参与 V_n（ε:A:ε 由本构 σ^u 给出，
    与 mech_init.py 的 Rayleigh 商约定一致）。
    """
    xy = xy.detach().requires_grad_(True)
    u_x = mech["u_x"](xy)
    u_y = mech["u_y"](xy)
    du_x = torch.autograd.grad(u_x.sum(), xy, create_graph=False)[0]
    du_y = torch.autograd.grad(u_y.sum(), xy, create_graph=False)[0]
    eps_xx = du_x[:, 0:1].detach()
    eps_yy = du_y[:, 1:2].detach()
    gamma_xy = (du_x[:, 1:2] + du_y[:, 0:1]).detach()
    return u_x.detach(), u_y.detach(), eps_xx, eps_yy, gamma_xy


def constitutive_stress(eps_xx, eps_yy, gamma_xy, e: torch.Tensor, cfg: HJBConfig):
    """平面应力本构：由应变算 σ^u。e 为 (N,1) 弹性模量场。"""
    nu = cfg.nu
    factor = e / (1.0 - nu * nu)
    sxx_u = factor * (eps_xx + nu * eps_yy)
    syy_u = factor * (eps_yy + nu * eps_xx)
    sxy_u = e / (2.0 * (1.0 + nu)) * gamma_xy
    return sxx_u, syy_u, sxy_u


# ---------------------------------------------------------------------------
# ω²（Rayleigh 商）、面积、目标函数 F —— 在积分点集上蒙特卡洛估计
# ---------------------------------------------------------------------------

def rayleigh_and_area(mech, phi_net, t_val: float, xy: torch.Tensor,
                      cfg: HJBConfig, chunk: int = 8192) -> Tuple[float, float]:
    """在给定点集上计算（对 t_val 切片的几何）：

        area = ∫S dx ≈ 域面积 × mean(S)
        R    = Σ S·(σ^u:ε) / Σ S·ρ̂·|u|²   （Rayleigh 商，振幅不变）

    力学网络冻结；材料场由 phi_net 在 t_val 切片给出。
    """
    s_sum = 0.0
    num = 0.0
    den = 0.0
    for s0 in range(0, xy.shape[0], chunk):
        xyb = xy[s0:s0 + chunk]
        with torch.no_grad():
            phi_val = eval_phi_at(phi_net, xyb, t_val)
            s, rho_hat, rho, e = fields_from_phi_val(phi_val, xyb, cfg)
        u_x, u_y, eps_xx, eps_yy, gamma_xy = frozen_strain(mech, xyb)
        sxx_u, syy_u, sxy_u = constitutive_stress(eps_xx, eps_yy, gamma_xy, e, cfg)
        energy = sxx_u * eps_xx + syy_u * eps_yy + sxy_u * gamma_xy   # ε:A:ε
        u_sq = u_x ** 2 + u_y ** 2
        s_sum += float(s.sum())
        num += float((s * energy).sum())
        den += float((s * rho_hat * u_sq).sum())
    area = cfg.area * s_sum / xy.shape[0]
    r_ray = num / max(den, 1e-30)
    return r_ray, area


def objective(omega2: float, area: float, cfg: HJBConfig) -> float:
    """F = −ω² + (1/2α)(∫S dx − C)²。"""
    return -omega2 + 0.5 / cfg.alpha * (area - cfg.v_target) ** 2


# ---------------------------------------------------------------------------
# 法向速度 V_n（算一次后固定，说明文档 §2、§5）
# ---------------------------------------------------------------------------

def compute_vn(mech, phi_ref, xy: torch.Tensor, cfg: HJBConfig,
               omega2: float, area: float, chunk: int = 8192):
    """V_n = ε:A:ε − ω²·ρ·|u|² − (1/α)(∫S dx − C)，逐点常数（(N,1)）。

    返回 (vn, energy, kinetic) 三个 (N,1) 张量（后两个供出图/诊断）。
    孔洞区内 E≈0、ρ≈0，前两项自然消失，只剩罚项（说明文档 §5）。
    """
    penalty = (area - cfg.v_target) / cfg.alpha
    vn = torch.empty(xy.shape[0], 1, dtype=cfg.dtype)
    eng = torch.empty(xy.shape[0], 1, dtype=cfg.dtype)
    kin = torch.empty(xy.shape[0], 1, dtype=cfg.dtype)
    for s0 in range(0, xy.shape[0], chunk):
        xyb = xy[s0:s0 + chunk]
        with torch.no_grad():
            phi_val = eval_phi_at(phi_ref, xyb, cfg.t_current)
            s, rho_hat, rho, e = fields_from_phi_val(phi_val, xyb, cfg)
        u_x, u_y, eps_xx, eps_yy, gamma_xy = frozen_strain(mech, xyb)
        sxx_u, syy_u, sxy_u = constitutive_stress(eps_xx, eps_yy, gamma_xy, e, cfg)
        energy = sxx_u * eps_xx + syy_u * eps_yy + sxy_u * gamma_xy
        kinetic = rho * (u_x ** 2 + u_y ** 2)
        vn[s0:s0 + chunk] = energy - omega2 * kinetic - penalty
        eng[s0:s0 + chunk] = energy
        kin[s0:s0 + chunk] = kinetic
    return vn, eng, kin


# ---------------------------------------------------------------------------
# 采样（Sobol 低差异点池）
# ---------------------------------------------------------------------------

def sample_spacetime_pool(n: int, cfg: HJBConfig, seed: int):
    """(x,y,t) 3D Sobol 点池，t ∈ [t_current, t_new]。返回 (xy, t)。"""
    eng = SobolEngine(dimension=3, scramble=True, seed=seed)
    u = eng.draw(n).to(cfg.dtype)
    lo = torch.tensor([0.0, 0.0, cfg.t_current], dtype=cfg.dtype)
    hi = torch.tensor([cfg.lx, cfg.ly, cfg.t_new], dtype=cfg.dtype)
    xyt = lo + u * (hi - lo)
    return xyt[:, :2].contiguous(), xyt[:, 2:3].contiguous()


def sample_init_pool(n: int, cfg: HJBConfig, seed: int):
    """初始条件点池：(x,y) 2D Sobol，t 恒为 t_current。返回 (xy, t)。"""
    eng = SobolEngine(dimension=2, scramble=True, seed=seed)
    xy = eng.draw(n).to(cfg.dtype) * torch.tensor([cfg.lx, cfg.ly], dtype=cfg.dtype)
    t = torch.full((n, 1), cfg.t_current, dtype=cfg.dtype)
    return xy, t


def sample_interior_sobol(n: int, cfg: HJBConfig, seed: int) -> torch.Tensor:
    """域内 (x,y) 2D Sobol（诊断积分点集用）。"""
    eng = SobolEngine(dimension=2, scramble=True, seed=seed)
    return eng.draw(n).to(cfg.dtype) * torch.tensor([cfg.lx, cfg.ly], dtype=cfg.dtype)


def sample_init_band(n: int, cfg: HJBConfig, gen: torch.Generator) -> torch.Tensor:
    """初始条件边界带点（锚定 rim）：沿四边内侧窄带采样，返回 (n+4, 2)。

    与 sdf_init.sample_boundary_band 同策略：带宽取敏感带 4β=0.04，按边长比例
    分配（最大余数法），每边内隔点严格压在边上（d=0），外加 4 个角点。
    """
    edges = [("bottom", cfg.lx), ("top", cfg.lx), ("left", cfg.ly), ("right", cfg.ly)]
    perim = 2.0 * (cfg.lx + cfg.ly)
    band_width = 4.0 * cfg.beta

    raw = [n * L / perim for _, L in edges]
    counts = [int(r) for r in raw]
    rem = n - sum(counts)
    order = sorted(range(4), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:rem]:
        counts[i] += 1

    pts = []
    for (name, L), m in zip(edges, counts):
        if m == 0:
            continue
        s = torch.rand(m, 1, generator=gen, dtype=cfg.dtype) * L
        d = torch.rand(m, 1, generator=gen, dtype=cfg.dtype) * band_width
        d[0::2] = 0.0  # 隔点严格压在边上
        if name == "bottom":
            p = torch.cat([s, d], dim=1)
        elif name == "top":
            p = torch.cat([s, cfg.ly - d], dim=1)
        elif name == "left":
            p = torch.cat([d, s], dim=1)
        else:  # right
            p = torch.cat([cfg.lx - d, s], dim=1)
        pts.append(p)

    corners = torch.tensor(
        [[0.0, 0.0], [cfg.lx, 0.0], [0.0, cfg.ly], [cfg.lx, cfg.ly]],
        dtype=cfg.dtype,
    )
    return torch.cat(pts + [corners], dim=0)


def minibatch_idx(n_pool: int, n_batch: int, gen: torch.Generator) -> torch.Tensor:
    """从点池随机抽 minibatch 索引（有放回）。"""
    return torch.randint(0, n_pool, (n_batch,), generator=gen)


# ---------------------------------------------------------------------------
# 点集构建（采样 + 预计算 V_n / 初始条件目标）
# ---------------------------------------------------------------------------

def phi_ref_val_and_gradnorm(phi_ref, xy: torch.Tensor, cfg: HJBConfig,
                             chunk: int = 16384):
    """冻结 phi_ref 在 t_current 切片的值与空间梯度模长（分块，一阶自动微分）。

    返回 (phi0, gnorm)，均为 (N,1) detach 常数。gnorm 加 1e-30 保护。
    """
    phi0 = torch.empty(xy.shape[0], 1, dtype=cfg.dtype)
    gnorm = torch.empty(xy.shape[0], 1, dtype=cfg.dtype)
    for s0 in range(0, xy.shape[0], chunk):
        xyb = xy[s0:s0 + chunk].detach().requires_grad_(True)
        tt = torch.full((xyb.shape[0], 1), cfg.t_current, dtype=cfg.dtype)
        p = phi_ref(torch.cat([xyb, tt], dim=1))
        (g,) = torch.autograd.grad(p.sum(), xyb, create_graph=False)
        phi0[s0:s0 + chunk] = p.detach()
        gnorm[s0:s0 + chunk] = torch.sqrt((g ** 2).sum(dim=1, keepdim=True) + 1e-30).detach()
    return phi0, gnorm


def build_interior_set(xy: torch.Tensor, t: torch.Tensor, mech, phi_ref,
                       cfg: HJBConfig, omega2: float, area: float) -> InteriorSet:
    vn, _, _ = compute_vn(mech, phi_ref, xy, cfg, omega2, area)
    # 一阶制造解锚点：phi_ref(t0) + (t−t0)·V_n·|∇phi_ref|（预计算常数）
    if cfg.lam_anchor > 0.0:
        phi0, gnorm = phi_ref_val_and_gradnorm(phi_ref, xy, cfg)
        phi_taylor = phi0 + (t - cfg.t_current) * vn * gnorm
    else:
        phi_taylor = torch.zeros(xy.shape[0], 1, dtype=cfg.dtype)
    return InteriorSet(xy=xy, t=t, vn=vn, phi_taylor=phi_taylor)


def build_init_set(xy: torch.Tensor, t: torch.Tensor, xy_band: torch.Tensor,
                   phi_ref, cfg: HJBConfig) -> InitSet:
    """初始条件点集 = 内部 Sobol 点（权重 1）+ 边界带点（权重 w_ic_band）。"""
    t_band = torch.full((xy_band.shape[0], 1), cfg.t_current, dtype=cfg.dtype)
    xy_all = torch.cat([xy, xy_band], dim=0)
    t_all = torch.cat([t, t_band], dim=0)
    with torch.no_grad():
        phi0 = phi_ref(torch.cat([xy_all, t_all], dim=1))
    w = torch.cat([
        torch.ones(xy.shape[0], 1, dtype=cfg.dtype),
        torch.full((xy_band.shape[0], 1), cfg.w_ic_band, dtype=cfg.dtype),
    ], dim=0)
    return InitSet(xy=xy_all, t=t_all, phi0=phi0, w=w)


def build_point_sets(cfg: HJBConfig, mech, phi_ref, omega2: float, area: float,
                     seed_offset: int, interior_n: int, init_n: int,
                     init_band_n: int) -> PointSets:
    """构建一套点集（Adam 点池 / L-BFGS 固定集 / 验证集共用此入口）。"""
    gen = torch.Generator().manual_seed(cfg.seed + seed_offset)
    xy_i, t_i = sample_spacetime_pool(interior_n, cfg, cfg.seed + seed_offset + 1)
    xy_0, t_0 = sample_init_pool(init_n, cfg, cfg.seed + seed_offset + 2)
    xy_band = sample_init_band(init_band_n, cfg, gen)
    return PointSets(
        interior=build_interior_set(xy_i, t_i, mech, phi_ref, cfg, omega2, area),
        init=build_init_set(xy_0, t_0, xy_band, phi_ref, cfg),
    )


# ---------------------------------------------------------------------------
# 损失（加权和，不乘掩码 S）：L_r + L_0 (+ 可选 L_anchor)；λ_r = λ_0 = λ_anchor = 1
# ---------------------------------------------------------------------------

def hjb_residual_loss(phi, pts: InteriorSet, create_graph: bool = True):
    """(1) HJB 残差：r = ∂φ̃/∂t − V_n·|∇φ̃|（符号约定见文件头说明）。

    ∂φ̃/∂t 与 |∇φ̃|（仅空间梯度）由自动微分计算；V_n 为预计算常数。
    返回 (loss, r.detach())。
    """
    xyt = torch.cat([pts.xy, pts.t], dim=1).detach().requires_grad_(True)
    p = phi(xyt)
    (g,) = torch.autograd.grad(p.sum(), xyt, create_graph=create_graph)
    phi_t = g[:, 2:3]
    gnorm = torch.sqrt((g[:, :2] ** 2).sum(dim=1, keepdim=True) + 1e-30)
    r = phi_t - pts.vn * gnorm
    return (r ** 2).mean(), r.detach()


def init_loss(phi, pts: InitSet) -> torch.Tensor:
    """(2) 初始条件：t = t_current 切片锚定 phi_ref（逐点权重：边界带 ×w_ic_band）。"""
    pred = phi(torch.cat([pts.xy, pts.t], dim=1))
    return (pts.w * (pred - pts.phi0) ** 2).sum() / pts.w.sum()


def anchor_loss(phi, pts: InteriorSet) -> torch.Tensor:
    """(3) 制造解锚点：φ 锚定到一阶制造解 phi_ref + (t−t0)·V_n·|∇phi_ref|。

    HJB 残差只约束比值 φ_t/|∇φ| = V_n（常函数是残差为零的退化解），
    单靠初始条件无法把剖面钉在整个时间柱面上；锚点给出显式剖面目标。
    """
    pred = phi(torch.cat([pts.xy, pts.t], dim=1))
    return ((pred - pts.phi_taylor) ** 2).mean()


def compute_total_loss(phi, data: PointSets, cfg: HJBConfig,
                       create_graph: bool = True) -> Tuple[torch.Tensor, Dict]:
    """总损失 L = λ_r·L_r + λ_0·L_0 (+ λ_a·L_anchor)。返回 (total, parts)。"""
    l_r, _ = hjb_residual_loss(phi, data.interior, create_graph)
    l_0 = init_loss(phi, data.init)
    total = cfg.lam_r * l_r + cfg.lam_0 * l_0
    parts = {
        "r": float(l_r.detach()),
        "init": float(l_0.detach()),
    }
    if cfg.lam_anchor > 0.0:
        l_a = anchor_loss(phi, data.interior)
        total = total + cfg.lam_anchor * l_a
        parts["anchor"] = float(l_a.detach())
    parts["total"] = float(total.detach())
    return total, parts


def evaluate_loss(phi, data: PointSets, cfg: HJBConfig) -> float:
    """验证/诊断用：只取总损失值（一阶导不建二阶图，省内存）。"""
    total, _ = compute_total_loss(phi, data, cfg, create_graph=False)
    return float(total.detach())


# ---------------------------------------------------------------------------
# 阶段 A：Adam（点池 + 每步 minibatch 重采，余弦退火）
# ---------------------------------------------------------------------------

def train_adam(phi, phi_ref, pools: PointSets, cfg: HJBConfig,
               gen: torch.Generator, history: list):
    if cfg.adam_steps <= 0:
        print("[Adam] adam_steps=0，跳过 Adam 段，直接进入 L-BFGS")
        return
    opt = torch.optim.Adam(phi.parameters(), lr=cfg.lr_init)
    # 余弦退火带上限：lr 在 anneal 步内从 lr_init 降到 lr_final，之后保持 lr_final
    anneal = min(cfg.anneal_steps, cfg.adam_steps)
    ratio = cfg.lr_final / cfg.lr_init

    def lr_lambda(step: int) -> float:
        p = min(step, anneal) / max(anneal, 1)
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    n_i = pools.interior.xy.shape[0]
    n_0 = pools.init.xy.shape[0]

    print(f"[Adam] 开始：{cfg.adam_steps} 步，lr {cfg.lr_init}→{cfg.lr_final}"
          f"（余弦，退火上限 {anneal} 步，之后保持 {cfg.lr_final}），"
          f"每步 batch {cfg.batch_interior}+{cfg.batch_init}")
    t0 = time.time()
    for step in range(cfg.adam_steps):
        data = PointSets(
            interior=pools.interior.subset(minibatch_idx(n_i, cfg.batch_interior, gen)),
            init=pools.init.subset(minibatch_idx(n_0, cfg.batch_init, gen)),
        )
        total, parts = compute_total_loss(phi, data, cfg)

        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.adam_steps - 1:
            parts["step"] = step
            parts["lr"] = sched.get_last_lr()[0]
            history.append(parts)
            anc = f" anchor {parts['anchor']:.3e}" if "anchor" in parts else ""
            print(f"[Adam] step {step:5d} | total {parts['total']:.4e} | "
                  f"r {parts['r']:.3e} init {parts['init']:.3e}{anc} | "
                  f"lr {parts['lr']:.2e} | {time.time()-t0:.0f}s")

        if step % cfg.fig_every == 0 or step == cfg.adam_steps - 1:
            save_snapshot(phi, phi_ref, cfg,
                          os.path.join(cfg.fig_dir, f"adam_step{step:06d}.png"),
                          title=f"Adam step {step}")
    print(f"[Adam] 结束，用时 {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# 阶段 B：L-BFGS 精调（固定点集，分块，优化器跨块复用，验证集只观察）
# ---------------------------------------------------------------------------

def train_lbfgs(phi, phi_ref, train_data: PointSets, val_data: PointSets,
                cfg: HJBConfig, history: list):
    params = list(phi.parameters())
    best_val = math.inf
    prev_val: Optional[float] = None
    best_state = copy.deepcopy(phi.state_dict()) if cfg.lbfgs_rollback else None
    stall = 0

    print(f"[L-BFGS] 开始：≤{cfg.lbfgs_blocks} 块 × {cfg.lbfgs_max_iter} 次，强 Wolfe，"
          f"固定点集 {cfg.lbfgs_interior}+{cfg.lbfgs_init}")
    # 优化器全程只建一次：曲率历史（逆 Hessian 近似）跨块累积；
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
        total, _ = compute_total_loss(phi, train_data, cfg, create_graph=True)
        total.backward()
        return total

    t0 = time.time()
    for block in range(cfg.lbfgs_blocks):
        opt.step(closure)

        train_loss = evaluate_loss(phi, train_data, cfg)
        val_loss = evaluate_loss(phi, val_data, cfg)

        rel_improve = (float("nan") if prev_val is None
                       else (prev_val - val_loss) / max(abs(prev_val), 1e-30))
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            if cfg.lbfgs_rollback:
                best_state = copy.deepcopy(phi.state_dict())
            stall = 0
        else:
            stall += 1
        prev_val = val_loss

        history.append({"step": f"lbfgs_{block}", "total": train_loss, "val": val_loss})
        mark = " *best" if is_best else ""
        rel_str = "      —" if math.isnan(rel_improve) else f"{rel_improve:+.2e}"
        print(f"[L-BFGS] block {block:3d} | train {train_loss:.6e} | "
              f"val {val_loss:.6e} | rel_improve {rel_str}{mark} | "
              f"{time.time()-t0:.0f}s")

        if block % cfg.fig_every_blocks == 0 or block == cfg.lbfgs_blocks - 1:
            save_snapshot(phi, phi_ref, cfg,
                          os.path.join(cfg.fig_dir, f"lbfgs_block{block:03d}.png"),
                          title=f"L-BFGS block {block}")

        if (cfg.lbfgs_early_stop and stall >= cfg.early_stop_patience
                and rel_improve < cfg.early_stop_rtol):
            print(f"[L-BFGS] 早停：连续 {stall} 块验证损失相对改善 < {cfg.early_stop_rtol}")
            break

    if cfg.lbfgs_rollback:
        phi.load_state_dict(best_state)
        print(f"[L-BFGS] 结束，已回滚到验证损失最优的权重（best val = {best_val:.6e}），"
              f"用时 {time.time()-t0:.0f}s")
    else:
        print(f"[L-BFGS] 结束，保留最终权重（不回滚；best val = {best_val:.6e} 仅供参考），"
              f"用时 {time.time()-t0:.0f}s")
    return best_val


# ---------------------------------------------------------------------------
# 存图
# ---------------------------------------------------------------------------

@torch.no_grad()
def _grid_xy(cfg: HJBConfig) -> torch.Tensor:
    """存图/诊断用规则网格，(fig_nx*fig_ny, 2)，indexing='xy' 展平。"""
    xs = torch.linspace(0.0, cfg.lx, cfg.fig_nx, dtype=cfg.dtype)
    ys = torch.linspace(0.0, cfg.ly, cfg.fig_ny, dtype=cfg.dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")   # (ny, nx)
    return torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)


def _panels_plot(panels, path: str, title: str, cfg: HJBConfig,
                 contours: Optional[List[Tuple[torch.Tensor, str, str]]] = None,
                 contour_ax: int = 0):
    """通用多面板 pcolormesh 存图。panels: [(Z(ny,nx) numpy, name, cmap)], ..."""
    xs = torch.linspace(0.0, cfg.lx, cfg.fig_nx).numpy()
    ys = torch.linspace(0.0, cfg.ly, cfg.fig_ny).numpy()
    X, Y = torch.meshgrid(torch.as_tensor(xs), torch.as_tensor(ys), indexing="xy")
    Xn, Yn = X.numpy(), Y.numpy()

    ncol = min(3, len(panels))
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 2.6 * nrow),
                             constrained_layout=True)
    axes = [axes] if nrow * ncol == 1 else list(axes.flat)
    for k, (ax, (Z, name, cmap)) in enumerate(zip(axes, panels)):
        pc = ax.pcolormesh(Xn, Yn, Z, cmap=cmap, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(name)
        fig.colorbar(pc, ax=ax, shrink=0.85)
        if k == contour_ax and contours:
            for Zc, color, label in contours:
                ax.contour(Xn, Yn, Zc, levels=[0.0], colors=[color],
                           linewidths=1.4)
            # 用 Line2D 代理建图例：零水平集可能不在图域内（空等值线），
            # 直接从 contour collection 取 handle 会崩
            handles = [Line2D([0], [0], color=c, lw=1.4, label=l)
                       for _, c, l in contours]
            ax.legend(handles=handles, loc="upper right", fontsize=8)
    for ax in axes[len(panels):]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_vn_figure(mech, phi_ref, cfg: HJBConfig, omega2: float, area: float,
                   path: str):
    """V_n 场及其两个分量（应变能密度、ω²ρ|u|² 动能密度）三联图。"""
    grid = _grid_xy(cfg)
    vn, eng, kin = compute_vn(mech, phi_ref, grid, cfg, omega2, area)
    shape = (cfg.fig_ny, cfg.fig_nx)
    lim = float(vn.abs().max())
    panels = [
        (vn.reshape(shape).numpy(), f"V_n (area−C penalty {(area-cfg.v_target)/cfg.alpha:+.3f})",
         "RdBu_r"),
        (eng.reshape(shape).numpy(), "strain energy ε:A:ε", "viridis"),
        ((omega2 * kin).reshape(shape).numpy(), "ω²ρ|u|²", "viridis"),
    ]
    # V_n 用对称色标突出正负
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 2.9), constrained_layout=True)
    xs = torch.linspace(0.0, cfg.lx, cfg.fig_nx).numpy()
    ys = torch.linspace(0.0, cfg.ly, cfg.fig_ny).numpy()
    X, Y = torch.meshgrid(torch.as_tensor(xs), torch.as_tensor(ys), indexing="xy")
    for ax, (Z, name, cmap) in zip(axes, panels):
        if cmap == "RdBu_r":
            pc = ax.pcolormesh(X.numpy(), Y.numpy(), Z, cmap=cmap, shading="auto",
                               vmin=-lim, vmax=lim)
        else:
            pc = ax.pcolormesh(X.numpy(), Y.numpy(), Z, cmap=cmap, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(name)
        fig.colorbar(pc, ax=ax, shrink=0.85)
    fig.suptitle(f"V_n field (fixed for this evolution) | ω² = {omega2:.6f}, "
                 f"area = {area:.4f}, C = {cfg.v_target}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def save_snapshot(phi, phi_ref, cfg: HJBConfig, path: str, title: str = ""):
    """四联图：φ(t_new) + 零水平集对比、S 演化前、S 演化后、ΔS。"""
    grid = _grid_xy(cfg)
    phi_before = eval_phi_at(phi_ref, grid, cfg.t_current)
    phi_after = eval_phi_at(phi, grid, cfg.t_new)
    s_before = 0.5 * (1.0 + torch.tanh(phi_before / (2.0 * cfg.beta)))
    s_after = 0.5 * (1.0 + torch.tanh(phi_after / (2.0 * cfg.beta)))
    shape = (cfg.fig_ny, cfg.fig_nx)
    ds = (s_after - s_before).reshape(shape).numpy()
    ds_lim = max(float(torch.tensor(ds).abs().max()), 1e-12)

    panels = [
        (phi_after.reshape(shape).numpy(), f"φ (t={cfg.t_new})", "viridis"),
        (s_before.reshape(shape).numpy(), f"S before (t={cfg.t_current})", "viridis"),
        (s_after.reshape(shape).numpy(), f"S after (t={cfg.t_new})", "viridis"),
        (ds, "ΔS = S_after − S_before", "RdBu_r"),
    ]
    contours = [
        (phi_before.reshape(shape).numpy(), "white", f"φ=0 @ t={cfg.t_current}"),
        (phi_after.reshape(shape).numpy(), "red", f"φ=0 @ t={cfg.t_new}"),
    ]
    # ΔS 面板用对称色标
    xs = torch.linspace(0.0, cfg.lx, cfg.fig_nx).numpy()
    ys = torch.linspace(0.0, cfg.ly, cfg.fig_ny).numpy()
    X, Y = torch.meshgrid(torch.as_tensor(xs), torch.as_tensor(ys), indexing="xy")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for k, (ax, (Z, name, cmap)) in enumerate(zip(axes.flat, panels)):
        if name.startswith("ΔS"):
            pc = ax.pcolormesh(X.numpy(), Y.numpy(), Z, cmap=cmap, shading="auto",
                               vmin=-ds_lim, vmax=ds_lim)
        else:
            pc = ax.pcolormesh(X.numpy(), Y.numpy(), Z, cmap=cmap, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(name)
        fig.colorbar(pc, ax=ax, shrink=0.85)
        if k == 0:
            for Zc, color, label in contours:
                ax.contour(X.numpy(), Y.numpy(), Zc, levels=[0.0],
                           colors=[color], linewidths=1.4)
            # Line2D 代理图例（避免零水平集不在图域内时取空 handle 崩溃）
            handles = [Line2D([0], [0], color=c, lw=1.4, label=l)
                       for _, c, l in contours]
            ax.legend(handles=handles, loc="upper right", fontsize=8)
    if title:
        fig.suptitle(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_loss_figure(adam_history: list, lbfgs_history: list, cfg: HJBConfig, path: str):
    """损失曲线：左图 Adam 各项分量，右图 L-BFGS 训练/验证总损失。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)

    if adam_history:
        steps = [h["step"] for h in adam_history]
        keys = [("total", "total"), ("r", "L_r (HJB)"), ("init", "L_0 (IC)")]
        if any("anchor" in h for h in adam_history):
            keys.append(("anchor", "L_anchor"))
        for key, label in keys:
            vals = [h.get(key, float("nan")) for h in adam_history]
            ax1.semilogy(steps, vals, label=label)
    ax1.set_xlabel("Adam step")
    ax1.set_ylabel("loss")
    ax1.set_title("Stage A (Adam)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    if lbfgs_history:
        blocks = list(range(len(lbfgs_history)))
        ax2.semilogy(blocks, [h["total"] for h in lbfgs_history], "o-", label="train")
        ax2.semilogy(blocks, [h["val"] for h in lbfgs_history], "s-", label="val")
    ax2.set_xlabel("L-BFGS block")
    ax2.set_ylabel("total loss")
    ax2.set_title("Stage B (L-BFGS)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_mech_fields_masked(mech, phi, cfg: HJBConfig, path: str):
    """最终输出（说明文档 §10）：位移/应力/应变乘 S(t_new) 后出图（孔洞区域置 0）。"""
    grid = _grid_xy(cfg)
    shape = (cfg.fig_ny, cfg.fig_nx)
    with torch.no_grad():
        s = 0.5 * (1.0 + torch.tanh(eval_phi_at(phi, grid, cfg.t_new) / (2.0 * cfg.beta)))
        # 位移与应力网络直接取值
        u_x = mech["u_x"](grid)
        u_y = mech["u_y"](grid)
        sxx = mech["sigma_xx"](grid)
        sxy = mech["sigma_xy"](grid)
        syy = mech["sigma_yy"](grid)

    # 应变需一阶自动微分（不能在 no_grad 下），分块避免一次性建图过大
    eps_xx = torch.empty(grid.shape[0], 1, dtype=cfg.dtype)
    eps_yy = torch.empty(grid.shape[0], 1, dtype=cfg.dtype)
    chunk = 8192
    for s0 in range(0, grid.shape[0], chunk):
        _, _, ex, ey, _ = frozen_strain(mech, grid[s0:s0 + chunk])
        eps_xx[s0:s0 + chunk] = ex
        eps_yy[s0:s0 + chunk] = ey

    with torch.no_grad():
        u_mag = torch.sqrt(u_x ** 2 + u_y ** 2)
        fields = [
            (u_mag * s, "|u|·S"), (sxx * s, "σ_xx·S"), (sxy * s, "σ_xy·S"),
            (syy * s, "σ_yy·S"), (eps_xx * s, "ε_xx·S"), (eps_yy * s, "ε_yy·S"),
        ]
        panels = [(f.reshape(shape).numpy(), name, "RdBu_r") for f, name in fields]
    _panels_plot(panels, path, f"mechanics fields × S (t={cfg.t_new})", cfg)


# ---------------------------------------------------------------------------
# 诊断（说明文档 §9）
# ---------------------------------------------------------------------------

def residual_rms(phi, data: PointSets, cfg: HJBConfig) -> float:
    """HJB 残差 RMS：sqrt(mean(r²))，在给定点集上。"""
    _, r = hjb_residual_loss(phi, data.interior, create_graph=False)
    return float(torch.sqrt((r ** 2).mean()))


def grad_norm_near_zero_level(phi, cfg: HJBConfig, chunk: int = 16384):
    """零水平集附近（|φ(t_new)| ≤ zero_band）|∇φ| 统计（评估是否需要重初始化）。"""
    grid = _grid_xy(cfg)
    norms = []
    vals = []
    for s0 in range(0, grid.shape[0], chunk):
        xyb = grid[s0:s0 + chunk]
        xyt = torch.cat([xyb, torch.full((xyb.shape[0], 1), cfg.t_new,
                                         dtype=cfg.dtype)], dim=1)
        xyt = xyt.detach().requires_grad_(True)
        p = phi(xyt)
        (g,) = torch.autograd.grad(p.sum(), xyt, create_graph=False)
        norms.append(g[:, :2].norm(dim=1).detach())
        vals.append(p.detach().squeeze(1))
    gn = torch.cat(norms)
    pv = torch.cat(vals)
    mask = pv.abs() <= cfg.zero_band
    if mask.sum() == 0:
        return None
    gnm = gn[mask]
    return {"mean": float(gnm.mean()), "min": float(gnm.min()),
            "max": float(gnm.max()), "n": int(mask.sum())}


def fem_cross_check(phi, cfg: HJBConfig):
    """演化后几何的 FEM ω² 对拍（说明文档 §9）：细网格 160×50，shift-invert 最小特征值。

    返回 (omega2_fem, volume_fem)。需要 scipy（fem_solver.py 的依赖）。
    """
    from fem_solver import FEMConfig, solve_eigen

    fem_cfg = FEMConfig(lx=cfg.lx, ly=cfg.ly, do_coarse=False)

    @torch.no_grad()
    def phi_fn(xy_np):
        xt = torch.as_tensor(xy_np, dtype=cfg.dtype)
        tt = torch.full((xt.shape[0], 1), cfg.t_new, dtype=cfg.dtype)
        return phi(torch.cat([xt, tt], dim=1)).squeeze(1).numpy()

    res = solve_eigen(fem_cfg, phi_fn, fem_cfg.nx_fine, fem_cfg.ny_fine,
                      label=f"HJB 演化后 (t={cfg.t_new})")
    return float(res.omega2[0]), float(res.volume)


def diagnose(phi, mech, train_data: PointSets, val_data: PointSets,
             cfg: HJBConfig, omega2_before: float, area_before: float,
             xy_int: torch.Tensor) -> Dict:
    """训练后自检（说明文档 §9），返回诊断字典（写入 hjb_state.json）。"""
    # 1. HJB 残差 RMS（训练集与独立验证集）
    rms_train = residual_rms(phi, train_data, cfg)
    rms_val = residual_rms(phi, val_data, cfg)

    # 2/3. 演化前后 F、ω²、面积（"演化后"的 ω² 为冻结力学网络 + 新几何的
    #       Rayleigh 商近似——真正的 ω² 更新属于下一次外层迭代的力学重解）
    f_before = objective(omega2_before, area_before, cfg)
    omega2_after, area_after = rayleigh_and_area(mech, phi, cfg.t_new, xy_int, cfg)
    f_after = objective(omega2_after, area_after, cfg)

    # 5. 零水平集附近 |∇φ|
    gn = grad_norm_near_zero_level(phi, cfg)

    print("\n===== 诊断（HJB 演化后自检） =====")
    print(f"HJB 残差 RMS：train = {rms_train:.3e} | val = {rms_val:.3e}")
    print(f"面积 ∫S dx：演化前 {area_before:.6f} → 演化后 {area_after:.6f} "
          f"（目标 C = {cfg.v_target}，偏差 {area_before-cfg.v_target:+.4f} → "
          f"{area_after-cfg.v_target:+.4f}）")
    print(f"ω²（PINN Rayleigh 商）：演化前 {omega2_before:.6f} → "
          f"演化后 {omega2_after:.6f}（冻结力学近似）")
    print(f"目标函数 F：演化前 {f_before:.6f} → 演化后 {f_after:.6f} "
          f"（V_n 取最速下降方向，F 应下降）")
    if gn:
        print(f"零水平集附近（|φ|≤{cfg.zero_band}，{gn['n']} 点）|∇φ|："
              f"mean = {gn['mean']:.4f} | min = {gn['min']:.4f} | max = {gn['max']:.4f}"
              f"（距离函数应 ≈ 1，偏离多则需重初始化）")

    diag = {
        "residual_rms_train": rms_train,
        "residual_rms_val": rms_val,
        "area_before": area_before,
        "area_after": area_after,
        "omega2_before": omega2_before,
        "omega2_after_rayleigh_frozen_mech": omega2_after,
        "F_before": f_before,
        "F_after": f_after,
        "grad_norm_near_zero_level": gn,
    }

    # 6. FEM 对拍（演化后几何）
    if cfg.fem_check:
        try:
            omega2_fem, volume_fem = fem_cross_check(phi, cfg)
            rel = abs(omega2_fem - omega2_after) / max(abs(omega2_fem), 1e-30)
            print(f"FEM 对拍（演化后几何，160×50）：ω²_FEM = {omega2_fem:.6f} | "
                  f"∫S = {volume_fem:.6f} | 与 PINN Rayleigh 相对偏差 {rel:.2e}")
            diag["omega2_fem_after"] = omega2_fem
            diag["area_fem_after"] = volume_fem
            diag["fem_vs_rayleigh_rel"] = rel
        except Exception as exc:  # FEM 对拍失败不阻断主流程
            print(f"[警告] FEM 对拍失败（{exc}），跳过。")
            diag["omega2_fem_after"] = None
    return diag


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(cfg: Optional[HJBConfig] = None):
    cfg = cfg or HJBConfig()
    cfg.phi_path = _resolve(cfg.phi_path)
    cfg.mech_dir = _resolve(cfg.mech_dir)
    cfg.phi_out = _resolve(cfg.phi_out)
    cfg.state_out = _resolve(cfg.state_out)
    cfg.fig_dir = _resolve(cfg.fig_dir)
    os.makedirs(os.path.dirname(cfg.phi_out), exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    t_start = time.time()

    print("=" * 72)
    print(f"HJB 单次演化：t ∈ [{cfg.t_current}, {cfg.t_new}]（dt = {cfg.dt}），"
          f"α = {cfg.alpha}（固定），C = {cfg.v_target}")
    print("=" * 72)

    # 1. 加载网络：力学冻结；SDF 双份（phi_ref 冻结锚点 / phi 可训练热启动）
    mech = load_frozen_mech(cfg)
    print(f"已加载冻结力学网络：{cfg.mech_dir}/*{cfg.mech_suffix}")
    phi_ref = load_phi(cfg, frozen=True)
    phi = load_phi(cfg, frozen=False)
    print(f"已加载 SDF 网络：{cfg.phi_path}（phi_ref 冻结锚定初始条件，phi 热启动训练）")

    # 2. 当前几何上的 ω²（Rayleigh 商）与面积（大 Sobol 积分点集，全脚本共用）
    xy_int = sample_interior_sobol(cfg.diag_n, cfg, cfg.seed + 1)
    omega2_before, area_before = rayleigh_and_area(mech, phi_ref, cfg.t_current,
                                                   xy_int, cfg)
    f_before = objective(omega2_before, area_before, cfg)
    print(f"当前几何：ω²(Rayleigh) = {omega2_before:.6f} | ∫S dx = {area_before:.6f} "
          f"| F = {f_before:.6f}")

    # 3. V_n 场：算一次后固定（说明文档 §2）。Adam 点池 / L-BFGS 固定集 / 验证集
    #    各自预计算 V_n（逐点常数，训练中以查表方式使用）。
    gen = torch.Generator().manual_seed(cfg.seed + 2)
    pools = build_point_sets(cfg, mech, phi_ref, omega2_before, area_before,
                             seed_offset=10, interior_n=cfg.pool_interior,
                             init_n=cfg.pool_init, init_band_n=cfg.pool_init_band)
    vn = pools.interior.vn
    print(f"V_n 场（{vn.shape[0]} 池点）：mean = {float(vn.mean()):+.4f} | "
          f"min = {float(vn.min()):+.4f} | max = {float(vn.max()):+.4f}")
    save_vn_figure(mech, phi_ref, cfg, omega2_before, area_before,
                   os.path.join(cfg.fig_dir, "vn_field.png"))

    train_data = build_point_sets(cfg, mech, phi_ref, omega2_before, area_before,
                                  seed_offset=110, interior_n=cfg.lbfgs_interior,
                                  init_n=cfg.lbfgs_init,
                                  init_band_n=cfg.lbfgs_init_band)
    val_data = build_point_sets(cfg, mech, phi_ref, omega2_before, area_before,
                                seed_offset=210, interior_n=cfg.val_interior,
                                init_n=cfg.val_init,
                                init_band_n=cfg.val_init_band)
    print(f"点集：Adam 池 {cfg.pool_interior}+{cfg.pool_init}+{cfg.pool_init_band}(带) | "
          f"L-BFGS {cfg.lbfgs_interior}+{cfg.lbfgs_init}+{cfg.lbfgs_init_band}(带) | "
          f"验证 {cfg.val_interior}+{cfg.val_init}+{cfg.val_init_band}(带)")

    # 初始状态图
    save_snapshot(phi, phi_ref, cfg,
                  os.path.join(cfg.fig_dir, "step000000_initial.png"),
                  title="initial (phi = phi_ref, untrained)")

    # 4. 阶段 A：Adam
    adam_history: list = []
    train_adam(phi, phi_ref, pools, cfg, gen, adam_history)

    # 5. 阶段 B：L-BFGS
    lbfgs_history: list = []
    train_lbfgs(phi, phi_ref, train_data, val_data, cfg, lbfgs_history)

    # 6. 收尾：损失曲线、最终状态图、力学场乘 S 出图
    save_loss_figure(adam_history, lbfgs_history, cfg,
                     os.path.join(cfg.fig_dir, "loss_history.png"))
    save_snapshot(phi, phi_ref, cfg, os.path.join(cfg.fig_dir, "final.png"),
                  title=f"final (t={cfg.t_new})")
    save_mech_fields_masked(mech, phi, cfg,
                            os.path.join(cfg.fig_dir, "mech_fields_masked.png"))

    # 7. 保存演化后 SDF 权重（纯 state_dict，与 networks.py 约定一致）
    torch.save(phi.state_dict(), cfg.phi_out)
    print(f"[保存] 演化后 SDF 网络权重 -> {cfg.phi_out}")

    # 8. 诊断 + 状态记录
    diag = diagnose(phi, mech, train_data, val_data, cfg,
                    omega2_before, area_before, xy_int)
    state = {
        "t_current": cfg.t_current,
        "dt": cfg.dt,
        "t_new": cfg.t_new,
        "alpha": cfg.alpha,
        "v_target": cfg.v_target,
        "phi_in": cfg.phi_path,
        "phi_out": cfg.phi_out,
        "seed": cfg.seed,
        "wall_time_s": round(time.time() - t_start, 1),
        **diag,
    }
    with open(cfg.state_out, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"[保存] 演化状态与诊断记录 -> {cfg.state_out}")
    print(f"[完成] HJB 单次演化结束，用时 {time.time()-t_start:.0f}s。"
          f"图片输出目录：{cfg.fig_dir}")


if __name__ == "__main__":
    main()
