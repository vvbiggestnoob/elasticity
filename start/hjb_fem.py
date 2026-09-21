# -*- coding: utf-8 -*-
"""
hjb_fem.py — HJB 方程的网格数值求解版本（PINN 版 hjb_step.py 的对照基准）

与 hjb_step.py 的关系：
  - 输入完全相同：冻结的 5 个力学网络（V_n 的唯一来源）+ SDF 网络 phi_init.pt
    （初始条件）；V_n 场、ω²（Rayleigh 商）、面积全部复用 hjb_step.py 的实现，
    逐点一致——两个版本唯一的差别是 HJB 方程的求解器。
  - PINN 版：训练 SDF 网络拟合 t ∈ [t_current, t_current+dt] 上的解（软损失）；
  - 本版：在 160×50 规则网格（与 FEM 细网格一致，hx = hy = 0.01）上用
    一阶 Godunov 迎风格式（水平集方法的经典离散；HJB 为双曲方程，不用 FEM）
    + 显式 Euler 时间推进求解同一初值问题：
        ∂φ/∂t = V_n·|∇φ|  （符号约定与 hjb_step.py 相同，见该文件头部说明）
        φ(t=0) = phi_ref(t=0 切片)
        边界：无显式边界条件（出流式）——边界节点用内部单侧差分，
        边界值由 PDE + 内部信息决定，零水平集可自由后退/外扩。
        （2026-09-08 起不再使用零法向梯度 ghost cell，与 PINN 版移除 L_b 一致。）
    时间子步由 CFL 条件自适应：dt_sub ≤ cfl · min(hx,hy) / max|V_n|。

输出（hjb_fem_results/）：
  - hjb_fem_sdf.npz     网格坐标、φ(0)、V_n、演化后 φ(t_new)
  - hjb_fem_state.json  标量记录（ω²、面积前后、子步数、对比指标）
  - hjb_fem_evolution.png  演化前后 φ / S / ΔS + 零水平集对比（与 PINN 版同版式）
  - hjb_fem_vs_pinn.png    若存在 weights/phi_iter1.pt（PINN 结果）则自动对照：
                           φ_FEM vs φ_PINN、|Δφ|、ΔS 分布图，并打印数值指标

全程 float64；网络部分用 torch（复用 hjb_step.py），网格推进用 numpy。
所有相对路径相对本脚本所在目录解析。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 无界面后端，只保存图片
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from networks import NetworkConfig, build_sdf_network
from hjb_step import (
    HJBConfig,
    _resolve,
    compute_vn,
    eval_phi_at,
    load_frozen_mech,
    load_phi,
    rayleigh_and_area,
    sample_interior_sobol,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 配置（几何/材料/演化参数继承 HJBConfig，保证与 PINN 版一致）
# ---------------------------------------------------------------------------

@dataclass
class HJBFEMConfig(HJBConfig):
    # 网格（与《实验设置与计算范围.md》§8 的 FEM 细网格一致）
    nx: int = 160
    ny: int = 50
    # CFL 安全系数：dt_sub ≤ cfl · min(hx,hy) / max|V_n|
    cfl: float = 0.9
    # 可选：PINN 版演化结果（存在则自动对照）
    phi_pinn_path: str = "weights/phi_iter1.pt"
    # 输出
    out_dir: str = "hjb_fem_results"

    @property
    def hx(self) -> float:
        return self.lx / self.nx

    @property
    def hy(self) -> float:
        return self.ly / self.ny


# ---------------------------------------------------------------------------
# 网格与初始/速度场
# ---------------------------------------------------------------------------

def build_grid(cfg: HJBFEMConfig):
    """规则网格节点坐标。返回 (xs, ys, X, Y, nodes)，X/Y (ny+1, nx+1)，nodes (N,2)。"""
    xs = np.linspace(0.0, cfg.lx, cfg.nx + 1)
    ys = np.linspace(0.0, cfg.ly, cfg.ny + 1)
    X, Y = np.meshgrid(xs, ys)                      # (ny+1, nx+1)
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)
    return xs, ys, X, Y, nodes


def godunov_grad_norm(phi: np.ndarray, vn: np.ndarray,
                      hx: float, hy: float) -> np.ndarray:
    """一阶 Godunov 迎风格式的 |∇φ|（逐点按 V_n 符号选 stencil）。

    方程 φ_t = V_n·|∇φ|（即 φ_t + F|∇φ| = 0，F = −V_n）：
      V_n < 0（F > 0）：|∇φ|² = max(D⁻ˣ,0)² + min(D⁺ˣ,0)² + max(D⁻ʸ,0)² + min(D⁺ʸ,0)²
      V_n > 0（F < 0）：|∇φ|² = min(D⁻ˣ,0)² + max(D⁺ˣ,0)² + min(D⁻ʸ,0)² + max(D⁺ʸ,0)²
    边界：无显式边界条件（出流式）——缺失的差分用内部单侧差分替代，
    边界值由 PDE + 内部信息决定，零水平集可自由后退/外扩。
    """
    # x 方向单侧差分；边界上缺失的一侧用内部单侧差分替代
    dxm = np.empty_like(phi)
    dxp = np.empty_like(phi)
    dxm[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / hx
    dxm[:, 0] = dxm[:, 1]                        # 左边界：用内部单侧导数
    dxp[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / hx
    dxp[:, -1] = dxp[:, -2]                      # 右边界：用内部单侧导数
    # y 方向同理
    dym = np.empty_like(phi)
    dyp = np.empty_like(phi)
    dym[1:, :] = (phi[1:, :] - phi[:-1, :]) / hy
    dym[0, :] = dym[1, :]                        # 下边界
    dyp[:-1, :] = (phi[1:, :] - phi[:-1, :]) / hy
    dyp[-1, :] = dyp[-2, :]                      # 上边界

    g_neg = np.sqrt(np.maximum(dxm, 0.0) ** 2 + np.minimum(dxp, 0.0) ** 2
                    + np.maximum(dym, 0.0) ** 2 + np.minimum(dyp, 0.0) ** 2)
    g_pos = np.sqrt(np.minimum(dxm, 0.0) ** 2 + np.maximum(dxp, 0.0) ** 2
                    + np.minimum(dym, 0.0) ** 2 + np.maximum(dyp, 0.0) ** 2)
    return np.where(vn > 0.0, g_pos, g_neg)


def evolve_hjb_grid(phi0: np.ndarray, vn: np.ndarray, dt: float,
                    hx: float, hy: float, cfl: float,
                    verbose: bool = True) -> Tuple[np.ndarray, int, float]:
    """显式 Euler + Godunov 空间离散推进 φ：φ ← φ + dt_sub·V_n·|∇φ|。

    子步数由 CFL 自适应：n_sub = ceil(max|V_n|·dt / (cfl·min(hx,hy)))。
    返回 (phi_new, n_sub, dt_sub)。
    """
    vmax = float(np.abs(vn).max())
    n_sub = max(1, int(np.ceil(vmax * dt / (cfl * min(hx, hy)))))
    dt_sub = dt / n_sub
    if verbose:
        print(f"[网格推进] max|V_n| = {vmax:.4f} | CFL：{n_sub} 个子步 × "
              f"dt_sub = {dt_sub:.2e}（上限 {cfl * min(hx, hy) / max(vmax, 1e-30):.2e}）")
    phi = phi0.copy()
    for _ in range(n_sub):
        phi = phi + dt_sub * vn * godunov_grad_norm(phi, vn, hx, hy)
    return phi, n_sub, dt_sub


# ---------------------------------------------------------------------------
# 指标与对比
# ---------------------------------------------------------------------------

def heaviside_np(phi: np.ndarray, beta: float) -> np.ndarray:
    """S = ½(1+tanh(φ/2β))（numpy 版）。"""
    return 0.5 * (1.0 + np.tanh(phi / (2.0 * beta)))


def grid_area(s: np.ndarray, cfg: HJBFEMConfig) -> float:
    """节点均值 × 域面积（网格口径；与 PINN 对照时双方同口径）。"""
    return cfg.area * float(s.mean())


def load_pinn_phi(cfg: HJBFEMConfig) -> Optional[torch.nn.Module]:
    """若存在 PINN 演化结果则加载（冻结），否则返回 None。"""
    path = _resolve(cfg.phi_pinn_path)
    if not os.path.exists(path):
        return None
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype)
    phi = build_sdf_network(net_cfg)
    phi.load_state_dict(torch.load(path, weights_only=True))
    phi.eval()
    phi.requires_grad_(False)
    return phi


def sync_time_with_pinn_state(cfg: HJBFEMConfig) -> bool:
    """若存在 PINN 演化记录（cfg.state_out，hjb_step.py 写出），以其 t_current/dt 为准。

    对照必须在同一时间区间 [t_current, t_current+dt] 上进行；PINN 结果文件
    （phi_iter1.pt）记录的是它那次运行的时间区间，可能与本脚本配置的默认值不同
    （例如 hjb_step.py 的 dt 被改过但 phi_iter1.pt 尚未重跑）。返回是否发生同步。
    """
    path = _resolve(cfg.state_out)
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        st = json.load(f)
    t_cur, dt = float(st["t_current"]), float(st["dt"])
    if t_cur != cfg.t_current or dt != cfg.dt:
        print(f"[同步] PINN 记录 {cfg.state_out}：t_current={t_cur}, dt={dt}，"
              f"与当前配置（{cfg.t_current}, {cfg.dt}）不同，以 PINN 记录为准。")
        cfg.t_current, cfg.dt = t_cur, dt
    return True


def compare_with_pinn(phi_fem: np.ndarray, phi_pinn_net, nodes_t: torch.Tensor,
                      phi0: np.ndarray, cfg: HJBFEMConfig) -> Dict:
    """在同一网格上对比 FEM 网格解与 PINN 网络解（t = t_new 切片）。"""
    phi_pinn = eval_phi_at(phi_pinn_net, nodes_t, cfg.t_new).squeeze(1).numpy()
    phi_pinn = phi_pinn.reshape(cfg.ny + 1, cfg.nx + 1)

    dphi = phi_fem - phi_pinn
    s_fem = heaviside_np(phi_fem, cfg.beta)
    s_pinn = heaviside_np(phi_pinn, cfg.beta)
    ds = s_fem - s_pinn
    band = np.abs(phi0) <= 4.0 * cfg.beta      # 敏感带（零水平集附近）

    metrics = {
        "dphi_max_abs": float(np.abs(dphi).max()),
        "dphi_mean_abs": float(np.abs(dphi).mean()),
        "dphi_max_abs_band": float(np.abs(dphi[band]).max()) if band.any() else None,
        "dS_max_abs": float(np.abs(ds).max()),
        "dS_mean_abs": float(np.abs(ds).mean()),
        "area_fem": grid_area(s_fem, cfg),
        "area_pinn": grid_area(s_pinn, cfg),
    }
    print("\n===== 与 PINN 版对照（同一网格、同一 V_n、同一初值） =====")
    print(f"|Δφ|：max = {metrics['dphi_max_abs']:.4e} | mean = {metrics['dphi_mean_abs']:.4e} | "
          f"敏感带(|φ₀|≤4β) max = {metrics['dphi_max_abs_band']:.4e}")
    print(f"|ΔS|：max = {metrics['dS_max_abs']:.4e} | mean = {metrics['dS_mean_abs']:.4e}")
    print(f"面积（网格口径）：FEM {metrics['area_fem']:.6f} vs "
          f"PINN {metrics['area_pinn']:.6f} | 差 {metrics['area_fem']-metrics['area_pinn']:+.6f}")
    return {"metrics": metrics, "phi_pinn": phi_pinn, "dphi": dphi, "dS": ds}


# ---------------------------------------------------------------------------
# 存图
# ---------------------------------------------------------------------------

def save_evolution_figure(phi0, phi_fem, cfg: HJBFEMConfig, path: str):
    """四联图（与 PINN 版 save_snapshot 同版式）：φ(t_new)+零水平集对比、
    S 演化前、S 演化后、ΔS。"""
    s0 = heaviside_np(phi0, cfg.beta)
    s1 = heaviside_np(phi_fem, cfg.beta)
    ds = s1 - s0
    ds_lim = max(float(np.abs(ds).max()), 1e-12)

    xs = np.linspace(0.0, cfg.lx, cfg.nx + 1)
    ys = np.linspace(0.0, cfg.ly, cfg.ny + 1)
    X, Y = np.meshgrid(xs, ys)
    panels = [
        (phi_fem, f"φ_FEM (t={cfg.t_new})", "viridis", None),
        (s0, f"S before (t={cfg.t_current})", "viridis", None),
        (s1, f"S after (t={cfg.t_new})", "viridis", None),
        (ds, "ΔS = S_after − S_before", "RdBu_r", (-ds_lim, ds_lim)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for k, (ax, (Z, name, cmap, lim)) in enumerate(zip(axes.flat, panels)):
        kw = {"vmin": lim[0], "vmax": lim[1]} if lim else {}
        pc = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", **kw)
        ax.set_aspect("equal")
        ax.set_title(name)
        fig.colorbar(pc, ax=ax, shrink=0.85)
        if k == 0:
            ax.contour(X, Y, phi0, levels=[0.0], colors=["white"], linewidths=1.4)
            ax.contour(X, Y, phi_fem, levels=[0.0], colors=["red"], linewidths=1.4)
            handles = [
                Line2D([0], [0], color="white", lw=1.4, label=f"φ=0 @ t={cfg.t_current}"),
                Line2D([0], [0], color="red", lw=1.4, label=f"φ=0 @ t={cfg.t_new}"),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(f"HJB grid (Godunov) evolution, dt={cfg.dt}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_compare_figure(phi0, phi_fem, comp: Dict, cfg: HJBFEMConfig, path: str):
    """FEM 网格解 vs PINN 网络解：φ 对比（含零水平集）、|Δφ|、ΔS。"""
    phi_pinn, dphi, ds = comp["phi_pinn"], comp["dphi"], comp["dS"]
    xs = np.linspace(0.0, cfg.lx, cfg.nx + 1)
    ys = np.linspace(0.0, cfg.ly, cfg.ny + 1)
    X, Y = np.meshgrid(xs, ys)
    dphi_lim = max(float(np.abs(dphi).max()), 1e-12)
    ds_lim = max(float(np.abs(ds).max()), 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 4.6), constrained_layout=True)
    panels = [
        (phi_fem, "φ_FEM", "viridis", None),
        (phi_pinn, "φ_PINN", "viridis", None),
        (dphi, f"Δφ = φ_FEM − φ_PINN (max {dphi_lim:.2e})", "RdBu_r", (-dphi_lim, dphi_lim)),
        (ds, f"S_FEM − S_PINN (max {ds_lim:.2e})", "RdBu_r", (-ds_lim, ds_lim)),
    ]
    for k, (ax, (Z, name, cmap, lim)) in enumerate(zip(axes.flat, panels)):
        kw = {"vmin": lim[0], "vmax": lim[1]} if lim else {}
        pc = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", **kw)
        ax.set_aspect("equal")
        ax.set_title(name)
        fig.colorbar(pc, ax=ax, shrink=0.85)
        if k in (0, 1):
            ax.contour(X, Y, phi0, levels=[0.0], colors=["white"], linewidths=1.2)
            ax.contour(X, Y, Z, levels=[0.0], colors=["red"], linewidths=1.2)
    fig.suptitle(f"HJB grid vs PINN @ t={cfg.t_new} "
                 f"(white: phi=0 before, red: phi=0 after)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(cfg: Optional[HJBFEMConfig] = None):
    cfg = cfg or HJBFEMConfig()
    cfg.out_dir = _resolve(cfg.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    t_start = time.time()

    print("=" * 72)
    print(f"HJB 网格求解（Godunov 迎风，对照基准）：t ∈ [{cfg.t_current}, {cfg.t_new}]"
          f"（dt = {cfg.dt}），α = {cfg.alpha}，C = {cfg.v_target}")
    print(f"网格 {cfg.nx}×{cfg.ny}（{cfg.nx+1}×{cfg.ny+1} 节点，"
          f"hx = {cfg.hx}，hy = {cfg.hy}）")
    print("=" * 72)

    # 1. 输入：与 PINN 版完全相同（冻结力学网络 + SDF 网络）
    mech = load_frozen_mech(cfg)
    phi_ref = load_phi(cfg, frozen=True)
    print(f"已加载冻结力学网络与 SDF：{cfg.mech_dir}/*{cfg.mech_suffix}，{cfg.phi_path}")

    # 2. ω² 与面积（复用 hjb_step 的积分点集与实现，保证与 PINN 版一致）
    xy_int = sample_interior_sobol(cfg.diag_n, cfg, cfg.seed + 1)
    omega2, area_mc = rayleigh_and_area(mech, phi_ref, cfg.t_current, xy_int, cfg)
    print(f"当前几何：ω²(Rayleigh) = {omega2:.6f} | ∫S dx (MC) = {area_mc:.6f}")

    # 3. 网格上的初始 φ 与 V_n（同一实现，逐点一致）
    xs, ys, X, Y, nodes = build_grid(cfg)
    nodes_t = torch.as_tensor(nodes, dtype=cfg.dtype)
    phi0 = eval_phi_at(phi_ref, nodes_t, cfg.t_current).squeeze(1).numpy()
    phi0 = phi0.reshape(cfg.ny + 1, cfg.nx + 1)
    vn, _, _ = compute_vn(mech, phi_ref, nodes_t, cfg, omega2, area_mc)
    vn = vn.squeeze(1).numpy().reshape(cfg.ny + 1, cfg.nx + 1)
    print(f"V_n（网格）：mean = {vn.mean():+.4f} | min = {vn.min():+.4f} | "
          f"max = {vn.max():+.4f}")

    # 4. Godunov 迎风推进
    phi_fem, n_sub, dt_sub = evolve_hjb_grid(phi0, vn, cfg.dt, cfg.hx, cfg.hy, cfg.cfl)

    # 5. 网格口径的面积变化
    s0 = heaviside_np(phi0, cfg.beta)
    s1 = heaviside_np(phi_fem, cfg.beta)
    area0, area1 = grid_area(s0, cfg), grid_area(s1, cfg)
    print(f"面积（网格口径）：演化前 {area0:.6f} → 演化后 {area1:.6f} "
          f"（Δ = {area1-area0:+.6f}，目标 C = {cfg.v_target}）")

    # 6. 保存网格结果
    npz_path = os.path.join(cfg.out_dir, "hjb_fem_sdf.npz")
    np.savez(npz_path, xs=xs, ys=ys, phi0=phi0, vn=vn, phi_fem=phi_fem,
             t_current=cfg.t_current, dt=cfg.dt, t_new=cfg.t_new,
             omega2=omega2, n_sub=n_sub, dt_sub=dt_sub)
    print(f"[保存] 网格 SDF -> {npz_path}")

    state = {
        "t_current": cfg.t_current, "dt": cfg.dt, "t_new": cfg.t_new,
        "alpha": cfg.alpha, "v_target": cfg.v_target,
        "omega2_rayleigh": omega2, "area_mc_before": area_mc,
        "area_grid_before": area0, "area_grid_after": area1,
        "grid": {"nx": cfg.nx, "ny": cfg.ny, "hx": cfg.hx, "hy": cfg.hy},
        "n_sub": n_sub, "dt_sub": dt_sub, "cfl": cfg.cfl,
        "seed": cfg.seed, "wall_time_s": round(time.time() - t_start, 1),
    }

    # 7. 演化图（与 PINN 版同版式）
    save_evolution_figure(phi0, phi_fem, cfg,
                          os.path.join(cfg.out_dir, "hjb_fem_evolution.png"))

    # 8. 与 PINN 版对照（若存在 phi_iter1.pt）
    phi_pinn_net = load_pinn_phi(cfg)
    if phi_pinn_net is None:
        print(f"[提示] 未找到 PINN 结果 {cfg.phi_pinn_path}，仅输出网格解；"
              f"先运行 hjb_step.py 后再运行本脚本可自动对照。")
    else:
        comp = compare_with_pinn(phi_fem, phi_pinn_net, nodes_t, phi0, cfg)
        state["pinn_compare"] = comp["metrics"]
        save_compare_figure(phi0, phi_fem, comp, cfg,
                            os.path.join(cfg.out_dir, "hjb_fem_vs_pinn.png"))

    json_path = os.path.join(cfg.out_dir, "hjb_fem_state.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"[保存] 标量记录 -> {json_path}")
    print(f"[完成] 用时 {time.time()-t_start:.0f}s。输出目录：{cfg.out_dir}")


if __name__ == "__main__":
    main()
