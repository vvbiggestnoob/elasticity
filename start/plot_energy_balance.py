"""
plot_energy_balance.py — 画能量平衡场 Ae(u):e(u) − ω²ρ|u|²

加载 weights/ 下训练好的 5 个力学网络 + 冻结 SDF，在细网格上逐点计算：

    energy(x) = S · σ^u : ε(u)          （弹性能密度，Rayleigh 商分子的被积函数）
    mass(x)   = S · ρ̂ · |u|²            （惯性能密度，Rayleigh 商分母的被积函数）
    field(x)  = energy(x) − R · mass(x)

其中 ω² 用网络自身的 Rayleigh 商 R = Σenergy / Σmass 替代。
若网络是精确特征函数，field 处处 ≈ 0（且按构造其全域和恰为 0）。

输出：mech_figures/energy_balance_2d.png 与 energy_balance_3d.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

from mech_init import (
    MechInitConfig, _resolve, load_frozen_phi, material_fields,
    mech_quantities, constitutive_stress, _grid_points,
)
from networks import NetworkConfig, build_mechanics_networks, MECH_NET_NAMES
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    cfg = MechInitConfig()
    cfg.fig_nx, cfg.fig_ny = 321, 101          # 加密网格，图更光滑
    weights_dir = _resolve(cfg.weights_dir)

    phi_net = load_frozen_phi(cfg)
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly,
                            hidden_layers=cfg.hidden_layers,
                            hidden_width=cfg.hidden_width,
                            dtype=cfg.dtype)
    mech = build_mechanics_networks(net_cfg)
    for name in MECH_NET_NAMES:
        mech[name].load_state_dict(
            torch.load(os.path.join(weights_dir, f"{name}_init.pt"), weights_only=True))
    mech.eval()
    print(f"已加载力学网络权重：{weights_dir}")

    pts = _grid_points(cfg)
    s, rho_hat, rho, e = material_fields(phi_net, pts, cfg)
    q = mech_quantities(mech, pts, create_graph=False)
    sxx_u, syy_u, sxy_u = constitutive_stress(q, e, cfg)

    energy = (s * (sxx_u * q.eps_xx + syy_u * q.eps_yy + sxy_u * q.gamma_xy)).squeeze(1)
    mass = (s * rho_hat * (q.u_x ** 2 + q.u_y ** 2)).squeeze(1)

    R = float(energy.sum() / mass.sum())
    field = energy - R * mass
    print(f"Rayleigh 商 R = {R:.6f}（作为 ω²）")
    print(f"被积场全域和 Σfield·dA ≈ {float(field.sum()) * cfg.area / field.numel():.3e}"
          f"（精确特征函数应 ≈ 0）")
    print(f"|field| max = {float(field.abs().max()):.4e}")

    nx, ny = cfg.fig_nx, cfg.fig_ny
    X = pts[:, 0].reshape(nx, ny).numpy()
    Y = pts[:, 1].reshape(nx, ny).numpy()
    Z = field.detach().reshape(nx, ny).numpy()

    # 色标截断：重块负峰比基体大一个数量级，不截断会把基体压成空白。
    # 取 |field| 的 99% 分位数（重块面积占比 0.8%，恰被排除），重块区域饱和显示。
    import numpy as np
    vmax = float(np.percentile(np.abs(Z), 99.0))
    print(f"色标截断 vmax = {vmax:.4e}（|field| 的 99% 分位数；max = {abs(Z).max():.2e}，重块饱和）")

    out_dir = _resolve(cfg.fig_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 二维热力图 ----
    fig, ax = plt.subplots(figsize=(11, 3.6), constrained_layout=True)
    im = ax.pcolormesh(X, Y, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$S\,\sigma^u:\varepsilon - R\,S\hat\rho\,|u|^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title(rf"Energy balance  $Ae(u):e(u) - \omega^2\rho|u|^2$,  $\omega^2=R={R:.4f}$")
    p2d = os.path.join(out_dir, "energy_balance_2d.png")
    fig.savefig(p2d, dpi=150)
    plt.close(fig)

    # ---- 三维曲面图（高度同样截断，否则重块尖峰把其余区域压成平面） ----
    Zc = np.clip(Z, -vmax, vmax)
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Zc, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    rstride=2, cstride=2, linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("energy − R·mass")
    ax.view_init(elev=28, azim=-128)
    ax.set_title(rf"Energy balance  $Ae(u):e(u) - \omega^2\rho|u|^2$,  $\omega^2=R={R:.4f}$")
    p3d = os.path.join(out_dir, "energy_balance_3d.png")
    fig.savefig(p3d, dpi=150)
    plt.close(fig)

    print(f"已保存：\n  {p2d}\n  {p3d}")


if __name__ == "__main__":
    main()
