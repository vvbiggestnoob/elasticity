# -*- coding: utf-8 -*-
"""对比：同一几何上，当前 PINN+HJB 的 V_n 与 towangbt 传统方法（FEM + 分段常数
水平集）的单步速度场相差多少、差在哪里。

做法：取 exp01_bridge 第 36 步的输入状态（phi_iter35 @ t=0.175 + mech *_iter35），
  1. 当前侧：严格复演该步的 V_n = ε:A:ε − ω²·ρ·|u|² − (area−C)/α
     （PINN 力学，ω² = PINN Rayleigh 商 = 0.216834，α = 0.44380，含角部冻结置零）；
  2. 传统侧：把同一 SDF 几何栅格化为 160×50 的 0-1 分段常数 φ（φ>0 → 材料），
     按 towangbt fem_lame2d_E3_elem4_1.m 的原始公式计算：
       - FEM 广义特征值（ersatz：空洞 λ,μ×1e-2、ρ=0；中心 8h×8h 重块 ρ×100；
         左右边固支），振型质量归一化 uᵀMu = 1；
       - P_frequency = (2−φ)(−ε:A:ε + ω²ρ|u|²)（单元中心，u 取四节点平均）；
       - penalty_mu = clamp(2t·ω²/ΔV², 1e-6, 100)，t = 2；
       - 曲率 κ：原文第 466-509 行的迎风/中心混合差分格式；
       - d_k = −P_L = P_frequency + penalty_mu·ΔV + ββ·κ（β=0 最速下降，L_λ=0）。
     传统"增长速度" = −d_k（d_k 是 φ→2 即空洞方向的速度）。

两侧统一为"增长速度"（正 = 材料扩张）：当前 V_n 即增长速度（φ_t = V_n|∇φ|，
材料在 φ>0 侧）；传统侧取 −d_k。在零水平集（界面）点上逐点对比并分解：
频率项（PINN vs FEM）、罚项（标量）、曲率项（仅传统侧有）。

用法：python compare_vn_traditional.py [k]   （k 默认 36，用 phi_iter{k-1}/mech _iter{k-1}）
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
import torch

import hjb_step_eik as hjb_mod
from fem_solver import FEMConfig, build_mesh, q4_template, assemble, clamped_free_dofs
from networks import NetworkConfig, build_sdf_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(SCRIPT_DIR, "..", "exp01_bridge", "weights")
OUT_FIG = os.path.join(SCRIPT_DIR, "diag_figures", "compare_vn_traditional.png")

H = 0.01                 # towangbt 网格尺寸
T2 = 2.0                 # towangbt 的 t（罚系数 2t = 4）
BETA_BETA = 5e-4         # towangbt 的周长正则系数
MU_LO, MU_HI = 1e-6, 100.0   # towangbt 的 penalty_mu 上下限
V_FIX = 0.4
DT = 0.005


# ---------------------------------------------------------------------------
# 当前侧：复演第 k 步的 V_n
# ---------------------------------------------------------------------------
def current_vn(k: int, xy: np.ndarray):
    """返回 (vn, eng, kin, cfg, phi_ref, mech, state36)，xy 为 (N,2) numpy。"""
    with open(os.path.join(WEIGHTS, f"hjb_state_iter{k}.json"), encoding="utf-8") as f:
        st = json.load(f)
    cfg = hjb_mod.HJBConfig(t_current=st["t_current"], alpha=st["alpha"],
                            corner_freeze=0.08, dtype=torch.float64)
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype)
    mech_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly,
                             hidden_layers=cfg.mech_hidden_layers,
                             hidden_width=cfg.mech_hidden_width, dtype=cfg.dtype)
    phi_ref = build_sdf_network(net_cfg)
    phi_ref.load_state_dict(torch.load(
        os.path.join(WEIGHTS, f"phi_iter{k - 1}.pt"), weights_only=True))
    phi_ref.eval().requires_grad_(False)
    mech = hjb_mod.build_mechanics_networks(mech_cfg)
    for n in ("u_x", "u_y"):
        mech[n].load_state_dict(torch.load(
            os.path.join(WEIGHTS, f"{n}_iter{k - 1}.pt"), weights_only=True))
    mech.eval().requires_grad_(False)

    xt = torch.as_tensor(xy, dtype=cfg.dtype)
    vn, eng, kin = hjb_mod.compute_vn(mech, phi_ref, xt, cfg,
                                      st["omega2_before"], st["area_before"])
    return (vn.numpy(), eng.numpy(), kin.numpy(), cfg, phi_ref, st)


# ---------------------------------------------------------------------------
# 传统侧：towangbt 公式复刻
# ---------------------------------------------------------------------------
def towangbt_curvature(phi_mat: np.ndarray, h: float) -> np.ndarray:
    """原文第 466-509 行的差分格式（phi_mat: (ny,nx)，取值 1/2）。"""
    n2, n1 = phi_mat.shape
    gx = np.zeros_like(phi_mat)
    gx[:, :n1 // 2] = (phi_mat[:, 1:n1 // 2 + 1] - phi_mat[:, :n1 // 2]) / h
    gx[:, n1 // 2:] = (phi_mat[:, n1 // 2:] - phi_mat[:, n1 // 2 - 1:n1 - 1]) / h
    gy = np.zeros_like(phi_mat)
    gy[:n2 // 2, :] = (phi_mat[1:n2 // 2 + 1, :] - phi_mat[:n2 // 2, :]) / h
    gy[n2 // 2:, :] = (phi_mat[n2 // 2:, :] - phi_mat[n2 // 2 - 1:n2 - 1, :]) / h
    mod = np.sqrt(gx ** 2 + gy ** 2)
    m = mod > 1e-12
    ppx = np.where(m, gx / np.where(m, mod, 1.0), 0.0)
    ppy = np.where(m, gy / np.where(m, mod, 1.0), 0.0)
    gxx = np.zeros_like(phi_mat)
    gxx[:, 1:n1 // 2] = (ppx[:, 1:n1 // 2] - ppx[:, :n1 // 2 - 1]) / h
    gxx[:, n1 // 2:n1 - 1] = (ppx[:, n1 // 2 + 1:n1] - ppx[:, n1 // 2:n1 - 1]) / h
    gxx[:, 0] = ppx[:, 0] / h
    gxx[:, n1 - 1] = -ppx[:, n1 - 1] / h
    gyy = np.zeros_like(phi_mat)
    gyy[1:n2 // 2, :] = (ppy[1:n2 // 2, :] - ppy[:n2 // 2 - 1, :]) / h
    gyy[n2 // 2:n2 - 1, :] = (ppy[n2 // 2 + 1:n2, :] - ppy[n2 // 2:n2 - 1, :]) / h
    gyy[0, :] = ppy[0, :] / h
    gyy[n2 - 1, :] = -ppy[n2 - 1, :] / h
    return gxx + gyy


def traditional_dk(phi_ref, t_val: float):
    """在同一几何上按 towangbt 公式计算。返回含各分量的命名空间字典。"""
    fem_cfg = FEMConfig(do_coarse=False)
    mesh = build_mesh(160, 50, fem_cfg)          # h = 0.01，与 towangbt 一致

    # 栅格化：单元中心 SDF > 0 → 材料（towangbt φ=1），否则空洞（φ=2）
    xy_c = torch.as_tensor(mesh.centers, dtype=torch.float64)
    phi_c = hjb_mod.eval_phi_at(phi_ref, xy_c, t_val).numpy().ravel()
    phi_elem = np.where(phi_c > 0.0, 1.0, 2.0)
    # 中心重块：|x−0.8|<4h, |y−0.25|<4h 钉死为材料、ρ×100（原文第 206-207、578 行）
    in_blk = ((np.abs(mesh.centers[:, 0] - 0.8) < 4 * H)
              & (np.abs(mesh.centers[:, 1] - 0.25) < 4 * H))
    phi_elem[in_blk] = 1.0
    rho_e = np.where(phi_elem == 1.0, 1.0, 0.0)  # 空洞无质量
    rho_e[in_blk] = 100.0
    E_e = np.where(phi_elem == 1.0, 1.0, 1e-2)   # 空洞 λ,μ×1e-2（ersatz）

    K0, M0, C0, B_center = q4_template(mesh.hx, mesh.hy, fem_cfg.nu)
    K, M, edof = assemble(mesh, E_e, rho_e, K0, M0)
    free, _ = clamped_free_dofs(mesh, fem_cfg)
    vals, vecs = spla.eigsh(K[free][:, free], k=1, M=M[free][:, free],
                            sigma=0.0, which="LM")
    omega2 = float(vals[0])
    u = np.zeros(2 * mesh.nnode)
    u[free] = vecs[:, 0]
    u /= np.sqrt(u @ (M @ u))                    # 质量归一化 uᵀMu = 1
    if u[np.argmax(np.abs(u))] < 0:
        u = -u

    # 单元中心：应变能密度 se 与动能密度 ke（原文第 436-443 行）
    u_e = u[edof]                                # (E,8)
    eps = u_e @ B_center.T                       # (E,3)
    sig = E_e[:, None] * (eps @ C0.T)
    se = np.einsum("ij,ij->i", eps, sig)
    ux_n, uy_n = u[0::2], u[1::2]
    ux_c = ux_n[mesh.elem].mean(axis=1)
    uy_c = uy_n[mesh.elem].mean(axis=1)
    ke = omega2 * np.where(phi_elem == 1.0, 1.0, 0.0) * (ux_c ** 2 + uy_c ** 2)
    # 原文 P_frequency 的密度部分用结构密度 ρ（重块区也是 ρ=1，见第 442 行 rho）
    p_freq = np.where(phi_elem == 1.0, -se + ke, 0.0)   # (2−φ) 因子：空洞为 0

    area_mat = float((phi_elem == 1.0).sum() * mesh.hx * mesh.hy)
    dV = area_mat - V_FIX
    penalty_mu = float(np.clip(2.0 * T2 * omega2 / max(dV, 1e-12) ** 2,
                               MU_LO, MU_HI))

    phi_mat = phi_elem.reshape(mesh.ny, mesh.nx)
    kappa = towangbt_curvature(phi_mat, mesh.hx).ravel()

    # P_L = −P_freq − penalty_mu·ΔV − ββ·κ（L_λ=0）；d_k = −P_L（β_CG=0）
    p_l = -p_freq - penalty_mu * dV - BETA_BETA * kappa
    d_k = -p_l
    # 线搜索步长（原文第 571 行：max 取 ρ≤1 的单元，即非重块）
    nb = ~in_blk
    alpha_ls = 0.5 * mesh.hx / max(np.abs(p_l[nb]).max(), 1e-30)

    return dict(mesh=mesh, phi_elem=phi_elem, omega2=omega2, se=se, ke=ke,
                p_freq=p_freq, dV=dV, penalty_mu=penalty_mu, kappa=kappa,
                d_k=d_k, v_grow=-d_k, alpha_ls=alpha_ls, area_mat=area_mat)


# ---------------------------------------------------------------------------
# 界面点提取（φ=0 等值线）
# ---------------------------------------------------------------------------
def interface_points(phi_ref, t_val: float):
    xs = np.linspace(0.0, 1.6, 641)
    ys = np.linspace(0.0, 0.5, 201)
    X, Y = np.meshgrid(xs, ys)
    xy = torch.as_tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float64)
    Z = hjb_mod.eval_phi_at(phi_ref, xy, t_val).numpy().reshape(X.shape)
    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=[0.0])
    plt.close(fig)
    pts = [seg for seg in cs.allsegs[0] if len(seg) > 2]
    return np.concatenate(pts, axis=0)


# ---------------------------------------------------------------------------
def main(k: int = 36):
    # ---- 传统侧（先算，拿到几何与 FEM 量）----
    # 需要 phi_ref：从当前侧加载流程里取（这里先加载一次）
    with open(os.path.join(WEIGHTS, f"hjb_state_iter{k}.json"), encoding="utf-8") as f:
        st = json.load(f)
    net_cfg = NetworkConfig(dtype=torch.float64)
    phi_ref = build_sdf_network(net_cfg)
    phi_ref.load_state_dict(torch.load(
        os.path.join(WEIGHTS, f"phi_iter{k - 1}.pt"), weights_only=True))
    phi_ref.eval().requires_grad_(False)
    t_val = st["t_current"]

    trad = traditional_dk(phi_ref, t_val)

    # ---- 界面点 ----
    pts = interface_points(phi_ref, t_val)
    mesh = trad["mesh"]
    ix = np.clip((pts[:, 0] / mesh.hx).astype(int), 0, mesh.nx - 1)
    iy = np.clip((pts[:, 1] / mesh.hy).astype(int), 0, mesh.ny - 1)
    eidx = iy * mesh.nx + ix

    # 当前侧 V_n（含角部冻结置零）
    vn, eng, kin, cfg, _, _ = current_vn(k, pts)
    vn, eng, kin = vn.ravel(), eng.ravel(), kin.ravel()
    freq_pinn = eng - st["omega2_before"] * kin
    penalty_cur = (st["area_before"] - V_FIX) / st["alpha"]

    # 传统侧在界面点上（取所在单元的值）
    freq_fem = -(trad["p_freq"][eidx])           # 增长速度口径的频率项 = se−ke
    curv_trad = -BETA_BETA * trad["kappa"][eidx]  # 增长速度口径的曲率项
    penalty_trad = -trad["penalty_mu"] * trad["dV"]
    v_trad = freq_fem + penalty_trad + curv_trad

    # 角部冻结区内的点单独标记（当前 V_n 被置零，传统侧无此机制）
    cf = 0.08
    in_corner = ((np.minimum(pts[:, 0], 1.6 - pts[:, 0]) < cf)
                 & (np.minimum(pts[:, 1], 0.5 - pts[:, 1]) < cf))
    live = ~in_corner

    print("=" * 76)
    print(f"第 {k} 步输入状态：phi_iter{k - 1} @ t={t_val}，mech _iter{k - 1}")
    print(f"界面点 {len(pts)} 个（其中角部冻结区 {in_corner.sum()} 个，统计时剔除）")
    print("=" * 76)

    print("\n--- 标量对比 ---")
    print(f"ω² ：PINN Rayleigh = {st['omega2_before']:.6f} | "
          f"传统 FEM（ersatz 1e-2）= {trad['omega2']:.6f} | "
          f"相对差 {abs(trad['omega2'] - st['omega2_before']) / trad['omega2']:.2%}")
    print(f"材料面积：当前（平滑 S）= {st['area_before']:.4f} | "
          f"传统（0-1 栅格化）= {trad['area_mat']:.4f}")
    print(f"罚压力（增长速度口径，负=收缩）：当前 −(area−C)/α = {-penalty_cur:+.4f} | "
          f"传统 −penalty_mu·ΔV = {penalty_trad:+.4f}"
          f"（penalty_mu = {trad['penalty_mu']:.4g}，ΔV = {trad['dV']:+.4f}）")

    def stat(name, a, b, m):
        d = a - b
        corr = np.corrcoef(a[m], b[m])[0, 1]
        agree = np.mean((a[m] > 0) == (b[m] > 0))
        print(f"{name}：当前 mean={a[m].mean():+.4f} std={a[m].std():.4f} | "
              f"传统 mean={b[m].mean():+.4f} std={b[m].std():.4f} | "
              f"差 mean={d[m].mean():+.4f} rms={np.sqrt((d[m] ** 2).mean()):.4f} | "
              f"相关 {corr:.3f} | 符号一致率 {agree:.1%}")

    print("\n--- 界面点逐点对比（剔除角部冻结区）---")
    stat("频率项（se−ke）", freq_pinn, freq_fem, live)
    stat("总速度 V_n vs −d_k", vn, v_trad, live)
    print(f"曲率项（仅传统侧）：mean={curv_trad[live].mean():+.4f} "
          f"std={curv_trad[live].std():.4f} max|·|={np.abs(curv_trad[live]).max():.4f}")

    print("\n--- 单步位移量 ---")
    print(f"当前：V_n·dt，dt={DT} → mean {np.abs(vn[live]).mean() * DT:.2e}，"
          f"max {np.abs(vn[live]).max() * DT:.2e}（h={H}）")
    print(f"传统：归一化步长 α_ls = {trad['alpha_ls']:.3e}，内层步进至 >3 单元翻转"
          f"（每外层迭代界面约推进 1 个单元 ≈ h）")

    # ---- 画图：几何 / 频率项对比 / 总速度对比 / 散点 ----
    hcfg = hjb_mod.HJBConfig(t_current=st["t_current"], alpha=st["alpha"],
                             corner_freeze=0.08, dtype=torch.float64,
                             mech_dir=WEIGHTS, mech_suffix=f"_iter{k - 1}.pt")
    mech_full = hjb_mod.load_frozen_mech(hcfg)
    xy_e = torch.as_tensor(mesh.centers, dtype=torch.float64)
    vn_e, eng_e, kin_e = hjb_mod.compute_vn(mech_full, phi_ref, xy_e, hcfg,
                                            st["omega2_before"], st["area_before"])
    vn_e = vn_e.numpy().ravel()
    freq_pinn_e = (eng_e - st["omega2_before"] * kin_e).numpy().ravel()
    mat = trad["phi_elem"] == 1.0

    fig, axes = plt.subplots(2, 3, figsize=(17, 7.2), constrained_layout=True)
    xe = np.linspace(0, 1.6, 161)
    ye = np.linspace(0, 0.5, 51)

    def panel(ax, Z, title, symm=True):
        Zg = Z.reshape(mesh.ny, mesh.nx)
        pc = ax.pcolormesh(xe, ye, Zg, cmap="RdBu_r", shading="auto")
        if symm:
            v = np.nanmax(np.abs(Zg))
            if v > 0:
                pc.set_clim(-v, v)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        fig.colorbar(pc, ax=ax, shrink=0.85)

    panel(axes[0, 0], mat.astype(float), "几何（0-1 栅格化，材料=1）", symm=False)
    panel(axes[0, 1], np.where(mat, freq_pinn_e, np.nan), "当前频率项 se−ke（PINN，材料区）")
    panel(axes[0, 2], np.where(mat, -(trad["p_freq"]), np.nan), "传统频率项 se−ke（FEM，材料区）")
    panel(axes[1, 0], np.where(mat, vn_e, np.nan), "当前总速度 V_n（含罚、角部置零）")
    panel(axes[1, 1], np.where(mat, trad["v_grow"], np.nan), "传统总速度 −d_k（含罚与曲率）")
    ax = axes[1, 2]
    ax.scatter(freq_fem[live], freq_pinn[live], s=3, alpha=0.4)
    lim = np.abs(np.concatenate([freq_fem[live], freq_pinn[live]])).max()
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlabel("传统频率项（FEM）")
    ax.set_ylabel("当前频率项（PINN）")
    ax.set_title("界面点频率项散点")
    ax.grid(alpha=0.3)
    fig.suptitle(f"V_n 对比：当前 PINN+HJB vs towangbt 传统方法（第 {k} 步几何）")
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)
    print(f"\n[保存] 对比图 -> {OUT_FIG}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 36)
