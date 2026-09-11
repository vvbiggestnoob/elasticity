# -*- coding: utf-8 -*-
"""
compare_energy_balance.py — 能量平衡场 Ae(u):e(u) − ω²ρ|u|² 的 PINN 与 FEM 对拍

在 FEM 细网格（160×50）单元中心上逐点计算同一个能量平衡场

    field(x) = S·σ:ε − ω²·S·ρ̂·|u|²

  - PINN 版：weights/ 下 5 个力学网络 + 冻结 SDF；σ^u 由 u 网络自动微分经平面应力
    本构得到，ω² 取网络自身 Rayleigh 商 R（与 plot_energy_balance.py 的定义一致）；
  - FEM 版：fem_solver.solve_eigen 的第 1 阶模态；σ:ε 在单元中心取值（B(0,0)），
    |u|² 由四节点双线性形函数插值到单元中心，ω² 取 ω²_FEM（细网格最小特征值）。

注意：场是 u 的二次型，两个模态先各自按"单元中心网格上的质量积分 = 1"归一，
保证幅值可比（PINN 训练时 m≈1、FEM 精确 φᵀMφ=1，归一只是消除残差幅值差异）。
按构造两个场的全域和都 ≈ 0；场本身处处 ≈ 0 只对积分成立，逐点不必为零
（重块处动能主导为负、固支端应变能主导为正，是物理特征而非误差）。

输出：mech_figures/energy_balance_compare.png（PINN 版 / FEM 版 / 绝对误差 三联图）
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fem_solver import FEMConfig, load_sdf_fn, solve_eigen
from mech_init import (MechInitConfig, load_frozen_phi, material_fields,
                       mech_quantities, constitutive_stress)
from networks import NetworkConfig, build_mechanics_networks, MECH_NET_NAMES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


# ---------------------------------------------------------------------------
# PINN 版：力学网络 + 自动微分（定义与 plot_energy_balance.py 一致）
# ---------------------------------------------------------------------------
def pinn_field(centers: np.ndarray, cfg: MechInitConfig):
    """在 centers (N,2) 上评估 PINN 版场。返回 (field, R, m)，field 未归一。"""
    phi_net = load_frozen_phi(cfg)
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly,
                            hidden_layers=cfg.hidden_layers,
                            hidden_width=cfg.hidden_width,
                            dtype=cfg.dtype)
    mech = build_mechanics_networks(net_cfg)
    weights_dir = _resolve(cfg.weights_dir)
    for name in MECH_NET_NAMES:
        mech[name].load_state_dict(torch.load(
            os.path.join(weights_dir, f"{name}_init.pt"), weights_only=True))
    mech.eval()
    print(f"已加载力学网络权重：{weights_dir}")

    pts = torch.as_tensor(centers, dtype=cfg.dtype)
    s, rho_hat, rho, e = material_fields(phi_net, pts, cfg)
    q = mech_quantities(mech, pts, create_graph=False)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    energy = (s * (sxx_u * q.eps_xx + syy_u * q.eps_yy + sxy_u * q.gamma_xy)).squeeze(1)
    mass = (s * rho_hat * (q.u_x ** 2 + q.u_y ** 2)).squeeze(1)
    R = float(energy.sum() / mass.sum())
    field = (energy - R * mass).detach().numpy()
    m = float(mass.sum())          # 质量积分（未乘 dA 的版本，归一化时约掉）
    return field, R, m


# ---------------------------------------------------------------------------
# FEM 版：第 1 阶模态，单元中心 σ:ε 与插值 |u|²
# ---------------------------------------------------------------------------
def fem_field(res, omega2: float):
    """返回 (field, m)：field = S·σ:ε − ω²·S·ρ̂·|u|²（单元中心，未归一）。"""
    u = res.modes[:, 0]
    u_e = u[res.edof]                              # (E, 8)
    eps = u_e @ res.B_center.T                     # (E, 3)
    sig = res.E[:, None] * (eps @ res.C0.T)        # (E, 3)
    strain_energy = (sig * eps).sum(axis=1)        # σ:ε

    ux_c = u[0::2][res.mesh.elem].mean(axis=1)     # 双线性形函数在单元中心 = 四节点平均
    uy_c = u[1::2][res.mesh.elem].mean(axis=1)
    kinetic = omega2 * res.rho * (ux_c ** 2 + uy_c ** 2)

    field = res.S * strain_energy - kinetic
    m = float((res.rho * (ux_c ** 2 + uy_c ** 2)).sum())
    return field, m


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    fem_cfg = FEMConfig()
    mech_cfg = MechInitConfig()

    # FEM 细网格求解（提供单元中心网格 + 第 1 阶模态 + ω²_FEM）
    phi_fn = load_sdf_fn(fem_cfg)
    res = solve_eigen(fem_cfg, phi_fn, fem_cfg.nx_fine, fem_cfg.ny_fine, label="细网格")
    omega2_fem = float(res.omega2[0])
    dA = res.mesh.hx * res.mesh.hy

    # 两个版本的场（各自归一：单元中心网格上质量积分 = 1）
    field_pinn, R_pinn, m_pinn = pinn_field(res.mesh.centers, mech_cfg)
    field_fem, m_fem = fem_field(res, omega2_fem)
    field_pinn = field_pinn / (m_pinn * dA)
    field_fem = field_fem / (m_fem * dA)

    err = np.abs(field_pinn - field_fem)
    l2_rel = float(np.linalg.norm(field_pinn - field_fem)
                   / (np.linalg.norm(field_fem) + 1e-300))

    print("\n===== 定量对比 =====")
    print(f"PINN Rayleigh 商 R      = {R_pinn:.6f}")
    print(f"FEM 锚点 ω²_FEM         = {omega2_fem:.6f}"
          f"（相对差 {abs(R_pinn - omega2_fem) / omega2_fem:.3e}）")
    print(f"归一前质量积分：PINN m = {m_pinn * dA:.6f}，FEM m = {m_fem * dA:.6f}（目标 1）")
    print(f"归一后全域和 Σfield·dA：PINN = {field_pinn.sum() * dA:.3e}，"
          f"FEM = {field_fem.sum() * dA:.3e}（按构造 ≈ 0）")
    print(f"绝对误差：max = {err.max():.4e} | mean = {err.mean():.4e}")
    print(f"相对 L2 误差 ‖PINN−FEM‖/‖FEM‖ = {l2_rel:.3e}")

    # ---- 三联图：PINN 版 / FEM 版 / 绝对误差 ----
    nx, ny = res.mesh.nx, res.mesh.ny
    xe = np.linspace(0.0, fem_cfg.lx, nx + 1)
    ye = np.linspace(0.0, fem_cfg.ly, ny + 1)
    Z_pinn = field_pinn.reshape(ny, nx)
    Z_fem = field_fem.reshape(ny, nx)
    Z_err = err.reshape(ny, nx)

    # 色标截断：重块负峰比基体大一个数量级，不截断会把基体压成空白；
    # PINN / FEM 两版共用同一对称色标（99% 分位数，重块面积占比 0.8% 恰被排除）。
    vmax = float(np.percentile(np.abs(np.concatenate([field_pinn, field_fem])), 99.0))
    vmax_err = float(np.percentile(err, 99.0))
    print(f"色标截断：场 vmax = {vmax:.4e}（max = {max(abs(field_pinn).max(), abs(field_fem).max()):.2e}，"
          f"重块饱和）| 误差 vmax = {vmax_err:.4e}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.6), constrained_layout=True)
    panels = [
        (Z_pinn, "RdBu_r", -vmax, vmax,
         rf"PINN:  $S\sigma^u:\varepsilon - R\,S\hat\rho|u|^2$,  $R={R_pinn:.4f}$"),
        (Z_fem, "RdBu_r", -vmax, vmax,
         rf"FEM:  $S\sigma:\varepsilon - \omega^2 S\hat\rho|u|^2$,  $\omega^2_{{FEM}}={omega2_fem:.4f}$"),
        (Z_err, "hot", 0.0, vmax_err,
         rf"|PINN − FEM|  (max={err.max():.3e}, rel L2={l2_rel:.2e})"),
    ]
    for ax, (Z, cmap, vmin, vm, title) in zip(axes, panels):
        pc = ax.pcolormesh(xe, ye, Z, cmap=cmap, vmin=vmin, vmax=vm, shading="auto")
        ax.add_patch(plt.Rectangle(
            (fem_cfg.block_x[0], fem_cfg.block_y[0]),
            fem_cfg.block_x[1] - fem_cfg.block_x[0],
            fem_cfg.block_y[1] - fem_cfg.block_y[0],
            fill=False, edgecolor="red", lw=0.8, ls="--"))
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(pc, ax=ax, shrink=0.9)

    out = _resolve(os.path.join(mech_cfg.fig_dir, "energy_balance_compare.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n已保存：{out}")


if __name__ == "__main__":
    main()
