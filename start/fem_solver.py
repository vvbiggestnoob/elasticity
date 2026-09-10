# -*- coding: utf-8 -*-
"""
FEM 求解器实现（对应《FEM计算说明.md》，唯一权威说明；数值取自《实验设置与计算范围.md》）

角色：基准 / 锚点提供者。给定神经网络 SDF（默认加载 weights/phi_init.pt，取 t=0 切片），
用 Q4 平面应力单元求广义特征值问题

    K φ = ω² M φ

的最小若干阶特征值与参考模态（质量归一化 φᵀMφ = 1），供 PINN 训练锚定与事后对拍。

流程（与《FEM计算说明.md》§2~§6 一一对应）：
  1. SDF → 材料场：单元中心采样 φ → S = ½(1+tanh(φ/2β))；
     ρ_e = S_e·ρ̂_e（空洞无质量），E_e = S_e·E_solid + (1−S_e)·E_void（ersatz 插值）；
     ρ̂：重块区 100、其余 1（硬指示函数，与 mech_init.py 的约定一致）。
  2. Q4 双线性矩形单元，2×2 高斯积分，平面应力本构（ν=0.3）。
     均匀网格下所有单元的"单位模板"相同：K0/M0（E=1、ρ=1）只算一次，
     逐单元分别乘 E_e / ρ_e，组装用 COO 一次性完成。
  3. 边界条件：左 x=0、右 x=Lx 全固支（u_x=u_y=0，从系统中消去）；
     上、下边界自由（自然满足，无需处理）。
  4. shift-invert 的 eigsh（sigma=0）取最小 n_eig 阶特征值，模态质量归一化。
  5. 细网格 160×50 + 粗网格 80×25 → Richardson 外推 (4·ω²_细 − ω²_粗)/3。
  6. 画图：u_x、u_y、σ_xx、σ_xy、σ_yy 五个场均乘 H(SDF) 过滤空洞，
     叠加 φ=0 零等值线与重块轮廓；另画特征值分布图。

依赖：numpy + scipy.sparse + matplotlib（Agg 后端，只存图）+ torch（仅用于 SDF 网络前向）。
全程 float64。
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")  # 无界面后端，只保存图片
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from networks import NetworkConfig, build_sdf_network

# 所有相对路径相对本脚本所在目录解析，从任意工作目录运行均可
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


# ---------------------------------------------------------------------------
# 配置（全部数值取自《实验设置与计算范围.md》）
# ---------------------------------------------------------------------------
@dataclass
class FEMConfig:
    # 几何（§3）
    lx: float = 1.6
    ly: float = 0.5
    # 网格（§8）：细网格用于锚点，粗网格用于 Richardson 外推
    nx_fine: int = 160
    ny_fine: int = 50
    nx_coarse: int = 80
    ny_coarse: int = 25
    do_coarse: bool = True          # 是否算粗网格（Richardson 用）
    # 材料参数（§4）
    e_solid: float = 1.0
    e_void: float = 1e-6            # ersatz 空洞模量（避免刚度阵奇异）
    nu: float = 0.3                 # 泊松比，平面应力
    rho_base: float = 1.0           # 基体密度
    rho_block: float = 100.0        # 重块密度
    # 重块区域（§5，不可优化，硬指示函数）
    block_x: tuple = (0.76, 0.84)
    block_y: tuple = (0.21, 0.29)
    # SDF / Heaviside（§7）
    beta: float = 0.01              # S = ½(1+tanh(φ/2β))
    # 特征值求解（§4）：shift-invert 取最小 n_eig 阶
    n_eig: int = 6
    # 路径
    phi_path: str = "weights/phi_init.pt"
    fig_dir: str = "fem_figures"
    out_dir: str = "fem_results"
    dtype: torch.dtype = torch.float64


# ---------------------------------------------------------------------------
# SDF：加载网络权重，包装成 phi_fn(xy) 可调用对象（t = 0 切片）
# ---------------------------------------------------------------------------
def load_sdf_fn(cfg: FEMConfig) -> Callable[[np.ndarray], np.ndarray]:
    """加载 SDF 网络权重，返回 phi_fn：xy (N,2) numpy -> phi (N,) numpy。

    SDF 网络结构复用 networks.py 的 build_sdf_network()（3×64 tanh，输入 (x,y,t)，
    xy 通道内置归一化，float64）；初始化/对拍阶段 t 恒取 0。
    """
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype)
    phi = build_sdf_network(net_cfg)
    phi.load_state_dict(torch.load(_resolve(cfg.phi_path), weights_only=True))
    phi.eval()
    phi.requires_grad_(False)

    @torch.no_grad()
    def phi_fn(xy: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(xy, dtype=cfg.dtype)
        t = torch.zeros(xt.shape[0], 1, dtype=cfg.dtype)
        return phi(torch.cat([xt, t], dim=1)).squeeze(1).numpy()

    return phi_fn


def heaviside(phi_val: np.ndarray, beta: float) -> np.ndarray:
    """S = H(φ) = ½(1+tanh(φ/2β))。"""
    return 0.5 * (1.0 + np.tanh(phi_val / (2.0 * beta)))


# ---------------------------------------------------------------------------
# 网格：均匀矩形 Q4 网格
# ---------------------------------------------------------------------------
def build_mesh(nx: int, ny: int, cfg: FEMConfig) -> SimpleNamespace:
    """均匀矩形网格。

    节点编号：node = iy*(nx+1) + ix（x 方向优先），坐标 (ix*hx, iy*hy)；
    单元编号：elem = iy*nx + ix，四节点逆时针 [n00, n10, n11, n01]；
    自由度：节点 n -> (2n, 2n+1) = (u_x, u_y)。
    """
    hx, hy = cfg.lx / nx, cfg.ly / ny
    xs = np.linspace(0.0, cfg.lx, nx + 1)
    ys = np.linspace(0.0, cfg.ly, ny + 1)
    X, Y = np.meshgrid(xs, ys)                      # (ny+1, nx+1)
    node_xy = np.stack([X.ravel(), Y.ravel()], axis=1)

    n00 = np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)[None, :]   # (ny, nx)
    elem = np.stack([n00, n00 + 1, n00 + nx + 2, n00 + nx + 1], axis=-1)
    elem = elem.reshape(-1, 4)

    xc = (np.arange(nx) + 0.5) * hx
    yc = (np.arange(ny) + 0.5) * hy
    CX, CY = np.meshgrid(xc, yc)                    # (ny, nx)，与单元编号一致
    centers = np.stack([CX.ravel(), CY.ravel()], axis=1)

    return SimpleNamespace(nx=nx, ny=ny, hx=hx, hy=hy,
                           node_xy=node_xy, elem=elem, centers=centers,
                           nnode=node_xy.shape[0], nelem=elem.shape[0])


# ---------------------------------------------------------------------------
# 材料场（《FEM计算说明.md》§2）：单元中心采样 SDF，乘 H(SDF) 过滤空洞
# ---------------------------------------------------------------------------
def element_materials(phi_fn: Callable[[np.ndarray], np.ndarray],
                      centers: np.ndarray, cfg: FEMConfig):
    """返回 (S, rho, E)，均为 (E,)：ρ_e = S_e·ρ̂_e，E_e = S_e·E_solid + (1−S_e)·E_void。"""
    phi_c = phi_fn(centers)
    S = heaviside(phi_c, cfg.beta)
    in_blk = ((centers[:, 0] >= cfg.block_x[0]) & (centers[:, 0] <= cfg.block_x[1])
              & (centers[:, 1] >= cfg.block_y[0]) & (centers[:, 1] <= cfg.block_y[1]))
    rho_hat = np.where(in_blk, cfg.rho_block, cfg.rho_base)
    rho = S * rho_hat                                   # 空洞无质量
    E = S * cfg.e_solid + (1.0 - S) * cfg.e_void        # ersatz 插值
    return S, rho, E


# ---------------------------------------------------------------------------
# Q4 单元（2×2 高斯积分，平面应力）
# ---------------------------------------------------------------------------
# 自然坐标下四节点 (ξ, η) = (−1,−1), (1,−1), (1,1), (−1,1)
_XI_N = np.array([-1.0, 1.0, 1.0, -1.0])
_ET_N = np.array([-1.0, -1.0, 1.0, 1.0])


def plane_stress_C0(nu: float) -> np.ndarray:
    """E = 1 的平面应力本构矩阵 C0（单元刚度 = E_e · ∫ Bᵀ C0 B）。"""
    return np.array([[1.0, nu, 0.0],
                     [nu, 1.0, 0.0],
                     [0.0, 0.0, (1.0 - nu) / 2.0]]) / (1.0 - nu * nu)


def _q4_B(xi: float, et: float, hx: float, hy: float) -> np.ndarray:
    """自然坐标 (ξ,η) 处的应变-位移矩阵 B (3,8)。矩形单元：dx/dξ=hx/2, dy/dη=hy/2。"""
    dN_dx = 0.25 * _XI_N * (1.0 + _ET_N * et) * (2.0 / hx)
    dN_dy = 0.25 * _ET_N * (1.0 + _XI_N * xi) * (2.0 / hy)
    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dx
    B[1, 1::2] = dN_dy
    B[2, 0::2] = dN_dy
    B[2, 1::2] = dN_dx
    return B


def q4_template(hx: float, hy: float, nu: float):
    """均匀矩形单元的单位模板：K0（E=1）、M0（ρ=1），2×2 高斯积分。

    均匀网格下所有单元形状相同，K_e = E_e·K0、M_e = ρ_e·M0，
    模板只算一次；返回 (K0, M0, C0, B_center)，B_center 为单元中心处的 B（算应力用）。
    """
    gp = 1.0 / np.sqrt(3.0)
    C0 = plane_stress_C0(nu)
    detJ = hx * hy / 4.0
    K0 = np.zeros((8, 8))
    M0 = np.zeros((8, 8))
    for xi in (-gp, gp):
        for et in (-gp, gp):
            B = _q4_B(xi, et, hx, hy)
            Nv = 0.25 * (1.0 + _XI_N * xi) * (1.0 + _ET_N * et)
            N = np.zeros((2, 8))
            N[0, 0::2] = Nv
            N[1, 1::2] = Nv
            K0 += (B.T @ C0 @ B) * detJ             # 高斯权重 = 1
            M0 += (N.T @ N) * detJ                  # 一致质量阵
    return K0, M0, C0, _q4_B(0.0, 0.0, hx, hy)


# ---------------------------------------------------------------------------
# 组装（COO 一次性）与边界条件（消去法）
# ---------------------------------------------------------------------------
def assemble(mesh: SimpleNamespace, E_e: np.ndarray, rho_e: np.ndarray,
             K0: np.ndarray, M0: np.ndarray):
    """组装总体刚度阵 K、质量阵 M（csc）。返回 (K, M, edof)，edof (E,8) 为单元自由度。"""
    edof = np.empty((mesh.nelem, 8), dtype=np.int64)
    edof[:, 0::2] = 2 * mesh.elem
    edof[:, 1::2] = 2 * mesh.elem + 1

    rows = np.broadcast_to(edof[:, :, None], (mesh.nelem, 8, 8)).ravel()
    cols = np.broadcast_to(edof[:, None, :], (mesh.nelem, 8, 8)).ravel()
    Ke = (E_e[:, None, None] * K0[None]).ravel()
    Me = (rho_e[:, None, None] * M0[None]).ravel()

    ndof = 2 * mesh.nnode
    K = sp.coo_matrix((Ke, (rows, cols)), shape=(ndof, ndof)).tocsc()
    M = sp.coo_matrix((Me, (rows, cols)), shape=(ndof, ndof)).tocsc()
    return K, M, edof


def clamped_free_dofs(mesh: SimpleNamespace, cfg: FEMConfig):
    """左 x=0、右 x=Lx 全固支（u_x=u_y=0，消去）；上、下自由（自然满足）。

    返回 (free, fixed) 自由度索引。
    """
    x = mesh.node_xy[:, 0]
    tol = 1e-12
    fixed_nodes = np.where((x <= tol) | (x >= cfg.lx - tol))[0]
    fixed = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]))
    free = np.setdiff1d(np.arange(2 * mesh.nnode), fixed)
    return free, fixed


# ---------------------------------------------------------------------------
# 特征值求解（《FEM计算说明.md》§4）
# ---------------------------------------------------------------------------
def solve_eigen(cfg: FEMConfig, phi_fn: Callable[[np.ndarray], np.ndarray],
                nx: int, ny: int, label: str = "") -> SimpleNamespace:
    """在 nx×ny 网格上完成 材料场 → 组装 → 加 BC → shift-invert 特征求解。

    返回命名空间：mesh, S, rho, E, edof, free, fixed, omega2 (n_eig,), modes (ndof, n_eig)
    （modes 为完整自由度向量，固支自由度为 0；已质量归一化 φᵀMφ=1，符号约定为
    最大绝对值分量为正）。
    """
    t0 = time.time()
    mesh = build_mesh(nx, ny, cfg)
    S, rho, E = element_materials(phi_fn, mesh.centers, cfg)
    K0, M0, C0, B_center = q4_template(mesh.hx, mesh.hy, cfg.nu)
    K, M, edof = assemble(mesh, E, rho, K0, M0)
    free, fixed = clamped_free_dofs(mesh, cfg)

    Kff = K[free][:, free]
    Mff = M[free][:, free]
    # shift-invert：sigma=0，OP = K⁻¹M，其最大特征值对应最小 ω²
    vals, vecs = spla.eigsh(Kff, k=cfg.n_eig, M=Mff, sigma=0.0, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    # 质量归一化（eigsh 已按 M 归一，这里显式重做一遍保证 φᵀMφ=1）+ 符号约定
    ndof = 2 * mesh.nnode
    modes = np.zeros((ndof, cfg.n_eig))
    for i in range(cfg.n_eig):
        v = vecs[:, i]
        v = v / np.sqrt(v @ (Mff @ v))
        k = int(np.argmax(np.abs(v)))
        if v[k] < 0.0:
            v = -v
        modes[free, i] = v

    volume = float(S.sum() * mesh.hx * mesh.hy)   # ∫S dΩ（单元中心求和）
    print(f"[{label}] 网格 {nx}×{ny}（{mesh.nnode} 节点 / {ndof} 自由度，"
          f"固支 {fixed.size} 自由度）| S∈[{S.min():.4f},{S.max():.4f}] | "
          f"∫S dΩ = {volume:.6f} | ω²_1 = {vals[0]:.8f} | 用时 {time.time()-t0:.1f}s")

    return SimpleNamespace(mesh=mesh, S=S, rho=rho, E=E, C0=C0, B_center=B_center,
                           edof=edof, free=free, fixed=fixed,
                           omega2=vals, modes=modes, volume=volume)


# ---------------------------------------------------------------------------
# 场恢复：位移（节点）+ 应力（单元中心，B(0,0) 处 ε = B·u_e，σ = E_e·C0·ε）
# ---------------------------------------------------------------------------
def mode_fields(res: SimpleNamespace, cfg: FEMConfig, mode: int = 0) -> SimpleNamespace:
    """提取第 mode 阶模态的位移/应力场（未掩码；掩码在画图时乘 H(SDF)）。"""
    mesh = res.mesh
    u = res.modes[:, mode]
    ux = u[0::2].reshape(mesh.ny + 1, mesh.nx + 1)
    uy = u[1::2].reshape(mesh.ny + 1, mesh.nx + 1)

    u_e = u[res.edof]                               # (E, 8)
    eps = u_e @ res.B_center.T                      # (E, 3) = [ε_xx, ε_yy, γ_xy]
    sig = res.E[:, None] * (eps @ res.C0.T)         # (E, 3) = [σ_xx, σ_yy, σ_xy]
    return SimpleNamespace(ux=ux, uy=uy, sig=sig)


# ---------------------------------------------------------------------------
# 画图（《FEM计算说明.md》§5：显示场乘 H(SDF) 过滤空洞，叠加 φ=0 等值线）
# ---------------------------------------------------------------------------
def plot_fields(res: SimpleNamespace, phi_fn: Callable[[np.ndarray], np.ndarray],
                cfg: FEMConfig, eig_summary: str, path: str, mode: int = 0) -> None:
    """六联图：u_x、u_y、σ_xx、σ_xy、σ_yy（均乘 H(SDF)）+ 材料场 S（几何参考）。"""
    mesh = res.mesh
    nx, ny = mesh.nx, mesh.ny
    f = mode_fields(res, cfg, mode)

    # 节点上的 φ 与 S（位移掩码 + φ=0 等值线）
    phi_n = phi_fn(mesh.node_xy).reshape(ny + 1, nx + 1)
    S_n = heaviside(phi_n, cfg.beta)
    S_e = res.S.reshape(ny, nx)                     # 单元中心材料场（应力掩码）

    ux_m = f.ux * S_n
    uy_m = f.uy * S_n
    sxx_m = (f.sig[:, 0] * res.S).reshape(ny, nx)
    syy_m = (f.sig[:, 1] * res.S).reshape(ny, nx)
    sxy_m = (f.sig[:, 2] * res.S).reshape(ny, nx)

    Xn = mesh.node_xy[:, 0].reshape(ny + 1, nx + 1)
    Yn = mesh.node_xy[:, 1].reshape(ny + 1, nx + 1)
    xe = np.linspace(0.0, cfg.lx, nx + 1)           # 单元场用单元边坐标（对齐单元格）
    ye = np.linspace(0.0, cfg.ly, ny + 1)

    panels = [
        (ux_m, "node", r"$u_x$ (masked)"),
        (uy_m, "node", r"$u_y$ (masked)"),
        (S_e, "elem", r"$S = H(\phi)$"),
        (sxx_m, "elem", r"$\sigma_{xx}$ (masked)"),
        (sxy_m, "elem", r"$\sigma_{xy}$ (masked)"),
        (syy_m, "elem", r"$\sigma_{yy}$ (masked)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 6.4), constrained_layout=True)
    for ax, (Z, kind, name) in zip(axes.flat, panels):
        if kind == "node":
            pc = ax.pcolormesh(Xn, Yn, Z, cmap="RdBu_r", shading="gouraud")
        else:
            cmap = "viridis" if name.startswith("S") else "RdBu_r"
            pc = ax.pcolormesh(xe, ye, Z, cmap=cmap, shading="auto")
        if not name.startswith("S"):                # 有符号场：对称色标
            vmax = float(np.abs(Z).max())
            if vmax > 0.0:
                pc.set_clim(-vmax, vmax)
        # φ=0 零等值线（材料/空洞界面；全材料时 φ=0 即外边界，无内部等值线可画）
        if phi_n.min() < 0.0 < phi_n.max():
            ax.contour(Xn, Yn, phi_n, levels=[0.0], colors="k", linewidths=0.8)
        # 重块轮廓
        ax.add_patch(plt.Rectangle(
            (cfg.block_x[0], cfg.block_y[0]),
            cfg.block_x[1] - cfg.block_x[0], cfg.block_y[1] - cfg.block_y[0],
            fill=False, edgecolor="red", lw=0.8, ls="--"))
        ax.set_aspect("equal")
        ax.set_title(name)
        fig.colorbar(pc, ax=ax, shrink=0.85)
    fig.suptitle(f"FEM mode {mode + 1} | {eig_summary}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eigenvalues(res_fine: SimpleNamespace, res_coarse: Optional[SimpleNamespace],
                     omega2_rich: Optional[np.ndarray], cfg: FEMConfig, path: str) -> None:
    """特征值分布图：细网格前 n_eig 阶（stem），叠加粗网格与 Richardson 外推的 ω₁²。"""
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    k = np.arange(1, cfg.n_eig + 1)
    markerline, stemlines, baseline = ax.stem(k, res_fine.omega2, basefmt=" ")
    plt.setp(markerline, marker="o", color="C0")
    plt.setp(stemlines, color="C0")
    markerline.set_label(f"fine {cfg.nx_fine}×{cfg.ny_fine}")
    if res_coarse is not None:
        ax.plot(k, res_coarse.omega2, "s--", color="C1",
                label=f"coarse {cfg.nx_coarse}×{cfg.ny_coarse}")
    if omega2_rich is not None:
        ax.plot([1], [omega2_rich[0]], "*", color="C3", ms=15,
                label=rf"Richardson $\omega_1^2$ = {omega2_rich[0]:.6f}")
    for ki, v in zip(k, res_fine.omega2):
        ax.annotate(f"{v:.4f}", (ki, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("mode index")
    ax.set_ylabel(r"$\omega^2$")
    ax.set_title(rf"FEM eigenvalues,  $\omega_1^2$(fine) = {res_fine.omega2[0]:.6f}")
    ax.set_xticks(k)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 结果保存：npz（供 PINN 锚定 / 对拍加载）+ 文本摘要
# ---------------------------------------------------------------------------
def save_results(res_fine: SimpleNamespace, res_coarse: Optional[SimpleNamespace],
                 omega2_rich: Optional[np.ndarray], cfg: FEMConfig,
                 summary_lines: list) -> str:
    """保存 fem_result.npz（细网格结果为主）与 fem_summary.txt。返回 npz 路径。"""
    os.makedirs(cfg.out_dir, exist_ok=True)
    mesh = res_fine.mesh
    npz_path = os.path.join(cfg.out_dir, "fem_result.npz")
    np.savez(
        npz_path,
        # 标量锚点
        omega2_fine=res_fine.omega2,
        omega2_coarse=(res_coarse.omega2 if res_coarse is not None
                       else np.full(cfg.n_eig, np.nan)),
        omega2_richardson=(omega2_rich if omega2_rich is not None
                           else np.full(cfg.n_eig, np.nan)),
        # 细网格网格信息
        nx=mesh.nx, ny=mesh.ny, hx=mesh.hx, hy=mesh.hy, lx=cfg.lx, ly=cfg.ly,
        node_xy=mesh.node_xy, elem=mesh.elem, centers=mesh.centers,
        free_dofs=res_fine.free, fixed_dofs=res_fine.fixed,
        # 参考模态（完整自由度，质量归一化）与材料场
        modes=res_fine.modes,
        mode1_ux=res_fine.modes[0::2, 0].reshape(mesh.ny + 1, mesh.nx + 1),
        mode1_uy=res_fine.modes[1::2, 0].reshape(mesh.ny + 1, mesh.nx + 1),
        elem_S=res_fine.S, elem_rho=res_fine.rho, elem_E=res_fine.E,
        volume_S=res_fine.volume,
    )
    with open(os.path.join(cfg.out_dir, "fem_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    return npz_path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(cfg: Optional[FEMConfig] = None) -> SimpleNamespace:
    cfg = cfg or FEMConfig()
    cfg.phi_path = _resolve(cfg.phi_path)
    cfg.fig_dir = _resolve(cfg.fig_dir)
    cfg.out_dir = _resolve(cfg.out_dir)
    os.makedirs(cfg.fig_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("=" * 72)
    print("FEM 特征值求解（基准 / 锚点）")
    print(f"  域 [0,{cfg.lx}]x[0,{cfg.ly}] | beta={cfg.beta} | "
          f"E_solid={cfg.e_solid}, E_void={cfg.e_void}, nu={cfg.nu} | "
          f"rho: 基体 {cfg.rho_base} / 重块 {cfg.rho_block}")
    print(f"  SDF: {cfg.phi_path}")
    print(f"  边界：左 x=0、右 x={cfg.lx} 全固支；上、下自由")
    print("=" * 72)

    phi_fn = load_sdf_fn(cfg)

    # 细网格（锚点）
    res_fine = solve_eigen(cfg, phi_fn, cfg.nx_fine, cfg.ny_fine, label="细网格")

    # 粗网格 + Richardson 外推（《FEM计算说明.md》§6）
    res_coarse, omega2_rich = None, None
    if cfg.do_coarse:
        res_coarse = solve_eigen(cfg, phi_fn, cfg.nx_coarse, cfg.ny_coarse, label="粗网格")
        omega2_rich = (4.0 * res_fine.omega2 - res_coarse.omega2) / 3.0

    # 汇总打印
    lines = [
        "FEM 特征值求解结果",
        f"SDF: {cfg.phi_path}",
        f"细网格 {cfg.nx_fine}x{cfg.ny_fine} / 粗网格 {cfg.nx_coarse}x{cfg.ny_coarse}",
        "",
        "mode   omega2_fine        omega2_coarse      omega2_richardson",
    ]
    for i in range(cfg.n_eig):
        c = res_coarse.omega2[i] if res_coarse is not None else float("nan")
        r = omega2_rich[i] if omega2_rich is not None else float("nan")
        lines.append(f"{i + 1:4d}   {res_fine.omega2[i]:.10f}   {c:.10f}   {r:.10f}")
    lines += [
        "",
        f"omega2_FEM（细网格最小特征值）= {res_fine.omega2[0]:.10f}",
        f"Richardson 外推 (4*omega2_fine - omega2_coarse)/3 = "
        f"{omega2_rich[0] if omega2_rich is not None else float('nan'):.10f}",
        f"int(S) dOmega（细网格）= {res_fine.volume:.6f}",
    ]
    print("\n===== 特征值结果 =====")
    for ln in lines[4:]:
        print(ln)

    eig_summary = (rf"$\omega_1^2$ fine = {res_fine.omega2[0]:.6f}"
                   + (rf", coarse = {res_coarse.omega2[0]:.6f}"
                      rf", Richardson = {omega2_rich[0]:.6f}"
                      if res_coarse is not None else ""))

    # 画图：5 个场（u_x, u_y, σ_xx, σ_xy, σ_yy）+ 材料场 S；特征值分布
    fig_fields = os.path.join(cfg.fig_dir, "fem_fields_mode1.png")
    plot_fields(res_fine, phi_fn, cfg, eig_summary, fig_fields, mode=0)
    fig_eig = os.path.join(cfg.fig_dir, "fem_eigenvalues.png")
    plot_eigenvalues(res_fine, res_coarse, omega2_rich, cfg, fig_eig)

    npz_path = save_results(res_fine, res_coarse, omega2_rich, cfg, lines)

    print(f"\n[保存] 场图 -> {fig_fields}")
    print(f"[保存] 特征值图 -> {fig_eig}")
    print(f"[保存] 数值结果 -> {npz_path}")
    print(f"[保存] 文本摘要 -> {os.path.join(cfg.out_dir, 'fem_summary.txt')}")
    print("[完成]")

    return SimpleNamespace(fine=res_fine, coarse=res_coarse,
                           omega2_richardson=omega2_rich)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FEM 特征值基准求解（给定 SDF）")
    p.add_argument("--phi", default=FEMConfig.phi_path,
                   help="SDF 网络权重路径（默认 weights/phi_init.pt）")
    p.add_argument("--n-eig", type=int, default=FEMConfig.n_eig,
                   help="求解的最小特征值阶数（默认 6）")
    p.add_argument("--no-coarse", action="store_true",
                   help="跳过粗网格与 Richardson 外推")
    p.add_argument("--fig-dir", default=FEMConfig.fig_dir, help="图片输出目录")
    p.add_argument("--out-dir", default=FEMConfig.out_dir, help="数值结果输出目录")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(FEMConfig(phi_path=args.phi, n_eig=args.n_eig,
                   do_coarse=not args.no_coarse,
                   fig_dir=args.fig_dir, out_dir=args.out_dir))
