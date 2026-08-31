"""独立出图脚本：从 checkpoint 里取出 ux / uy / sdf 三个网络，画

  1) 位移场            u_x, u_y
  2) 四个一阶偏导      du_x/dx, du_x/dy, du_y/dx, du_y/dy
  3) 材料分布          phi -> Heaviside -> 0/1 材料指示 -> 密度 rho（重块强制有材料，密度可调）
  4) 目标量            q = A e(u):e(u) - omega^2 * rho * |u|^2

与 config.py / pinn_core.py 放在同一目录下运行：

    python figures.py                                  # 默认 outputs/model_final.pt
    python figures.py --rho-block 1                    # 重块密度改成 1（可改量）
    python figures.py --rho-mode smooth                # 用训练口径的 S_eff 代替 0/1 二值
    python figures.py --omega2 0.454663                # 手动指定 omega^2

口径说明（很重要，决定 q 的积分是不是 0）：
  训练时用的密度是 rho = S_eff * (1 + (gamma-1) * chi_block)，掩码是 S_eff^2，
  omega^2 = R_final 正是在这套口径下算出来的 Rayleigh 商。因此只有 rho_block 与训练的
  gamma_block 一致时，int S^2 q dA 才会是机器零；把 rho_block 改成别的值，q 的积分会随之
  偏离 0，这不是 bug，是换了分母。脚本每次都会把两种口径的 R 和 int q 一起打印出来。
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pinn_core as core


# ------------------------------------------------------------------ 场的计算

def make_grid(cfg, nx, ny):
    """节点网格（含边界），返回 (X, Y, XY_tensor)，X/Y 形状 (ny, nx)。"""
    xs = np.linspace(0.0, cfg.Lx, nx)
    ys = np.linspace(0.0, cfg.Ly, ny)
    X, Y = np.meshgrid(xs, ys)
    XY = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1))
    return X, Y, XY


def _grad1(f, xy):
    """一阶导（不建图，省内存）：f (N,1) -> (df/dx, df/dy)。"""
    g = torch.autograd.grad(f, xy, grad_outputs=torch.ones_like(f))[0]
    return g[:, 0:1], g[:, 1:2]


def eval_u_and_grads(nets, XY, chunk=40000):
    """u 与它的四个一阶偏导。分块求导，网格再密也不会爆内存。"""
    out = {k: [] for k in ("ux", "uy", "ux_x", "ux_y", "uy_x", "uy_y")}
    for i in range(0, XY.shape[0], chunk):
        p = XY[i:i + chunk].clone().requires_grad_(True)
        ux, uy = nets["ux"](p), nets["uy"](p)
        ux_x, ux_y = _grad1(ux, p)
        uy_x, uy_y = _grad1(uy, p)
        for k, v in zip(out, (ux, uy, ux_x, ux_y, uy_x, uy_y)):
            out[k].append(v.detach())
    return {k: torch.cat(v, dim=0) for k, v in out.items()}


def chi_block_sharp(XY, block):
    """重块的硬指示函数：区间内 1、区间外 0（就是 md 里写的强制有材料的那块）。"""
    x, y = XY[:, 0:1], XY[:, 1:2]
    inx = (x >= block[0]) & (x <= block[1])
    iny = (y >= block[2]) & (y <= block[3])
    return (inx & iny).to(XY.dtype)


@torch.no_grad()
def material(sdf, XY, cfg, rho_block, rho_solid=1.0, rho_mode="binary", block_mode="sharp"):
    """材料分布与密度。返回 dict，全部是 (N,1) 张量。

    phi      SDF 网络输出
    S        Heaviside: 0.5*(1+tanh(phi/eps))，连续版材料指示
    chi      重块指示函数（sharp = 硬矩形，smooth = 训练用的 tanh 版）
    mat      材料指示：binary -> 0/1（S>=0.5 即有材料），smooth -> S_eff 本身
             两种模式下重块处都强制为 1
    rho      mat * [rho_solid + (rho_block - rho_solid) * chi]
             即：有材料 rho_solid（默认 1）、无材料 0、重块 rho_block（默认 = cfg.gamma_block）
    """
    phi = sdf(XY)
    S = core.heaviside_tanh(phi, cfg.eps_heaviside)
    chi = (chi_block_sharp(XY, cfg.block) if block_mode == "sharp"
           else core.chi_block(XY, cfg.block, cfg.block_smooth_w))
    if rho_mode == "binary":
        mat = (S >= 0.5).to(S.dtype)
    else:
        mat = S
    mat = mat + (1.0 - mat) * chi                  # 重块区强制有材料
    rho = mat * (rho_solid + (rho_block - rho_solid) * chi)
    return dict(phi=phi, S=S, chi=chi, mat=mat, rho=rho)


def energy_density(g, cfg):
    """A e(u):e(u) = sigma(u):e(u)，平面应力，只用到 u 的一阶导。"""
    exx, eyy = g["ux_x"], g["uy_y"]
    exy = 0.5 * (g["ux_y"] + g["uy_x"])
    sxx, syy, sxy = core.stress_from_strain(exx, eyy, exy, cfg.E, cfg.nu)
    return sxx * exx + syy * eyy + 2.0 * sxy * exy, (exx, eyy, exy)


def trapz_weights(shape):
    """节点网格上的梯形求积权重（边 1/2、角 1/4），归一化成"加权平均"。

    不能直接用 np.mean：左右夹持边应变能最大、动能为零，等权平均会让 Rayleigh 商偏高几个百分点。
    """
    ny, nx = shape
    wx = np.ones(nx); wx[0] = wx[-1] = 0.5
    wy = np.ones(ny); wy[0] = wy[-1] = 0.5
    W = np.outer(wy, wx)
    return W / W.sum()


# ------------------------------------------------------------------ 自检

def check_derivatives(nets, cfg, n=200, h=1e-5):
    """自动微分的四个偏导 vs 中心差分（随机内点，离边界留 0.05）。相对误差应 ~1e-8 以下。"""
    gen = torch.Generator().manual_seed(0)
    xy = torch.tensor([0.05, 0.05]) + torch.rand(n, 2, generator=gen) * \
        torch.tensor([cfg.Lx - 0.1, cfg.Ly - 0.1])
    g = eval_u_and_grads(nets, xy)
    ex, ey = torch.tensor([[h, 0.0]]), torch.tensor([[0.0, h]])
    out = {}
    with torch.no_grad():
        for name in ("ux", "uy"):
            for key, d in ((f"{name}_x", ex), (f"{name}_y", ey)):
                fd = (nets[name](xy + d) - nets[name](xy - d)) / (2.0 * h)
                out[key] = float((g[key] - fd).abs().max()) / (float(fd.abs().max()) + 1e-300)
    return out


def rayleigh_train_gauge(nets, sdf, cfg, nx, ny):
    """训练口径（S_eff^mask_pow_ray 掩码 + gamma_block 密度）下的 Rayleigh 商。

    omega^2 就是这么定出来的，所以拿它做求积收敛检查最直接：换个网格若数值明显变了，
    说明报出来的 omega^2 里含求积误差，而不是网络的问题。
    """
    X, _, XY = make_grid(cfg, nx, ny)
    g = eval_u_and_grads(nets, XY)
    en, _ = energy_density(g, cfg)
    _, _, _, S_eff, rho = core.material_fields_full(sdf, XY, cfg)
    MR = S_eff ** cfg.mask_pow_ray
    u2 = g["ux"] ** 2 + g["uy"] ** 2
    W = trapz_weights(X.shape)
    f = lambda t: float(np.sum(W * t.squeeze(1).numpy().reshape(X.shape)))
    return f(MR * en) / (f(MR * rho * u2) + 1e-300)


# ------------------------------------------------------------------ 绘图

def _add_block(ax, block):
    ax.add_patch(Rectangle((block[0], block[2]), block[1] - block[0], block[3] - block[2],
                           fill=False, edgecolor="k", lw=1.0, ls="--"))


def _sym_panel(fig, ax, X, Y, F, title, block, clip_pct=None):
    """对称色标热图。clip_pct 给定时用该分位数做稳健色标（重块处的尖峰不至于压平其余结构）。"""
    vmax = float(np.nanmax(np.abs(F))) + 1e-30
    v = vmax
    clipped = False
    if clip_pct is not None:
        vr = float(np.nanpercentile(np.abs(F), clip_pct)) + 1e-30
        if vmax / vr > 3.0:
            v, clipped = vr, True
    pc = ax.pcolormesh(X, Y, F, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    fig.colorbar(pc, ax=ax, extend="both" if clipped else "neither")
    _add_block(ax, block)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title + (f"\ncolour clipped at $\\pm${v:.3g} (true $\\pm${vmax:.3g})"
                          if clipped else f"\nmax|.| = {vmax:.4g}"), fontsize=9)


def fig_displacement(X, Y, UX, UY, block, path):
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), constrained_layout=True)
    _sym_panel(fig, axes[0], X, Y, UX, r"$u_x$", block)
    _sym_panel(fig, axes[1], X, Y, UY, r"$u_y$", block)
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_gradients(X, Y, G, block, path):
    keys = [("ux_x", r"$\partial u_x/\partial x$"), ("ux_y", r"$\partial u_x/\partial y$"),
            ("uy_x", r"$\partial u_y/\partial x$"), ("uy_y", r"$\partial u_y/\partial y$")]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 6.0), constrained_layout=True)
    for ax, (k, t) in zip(axes.ravel(), keys):
        _sym_panel(fig, ax, X, Y, G[k], t, block)
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_material(X, Y, M, block, eps, path):
    """四联图：phi（带 phi=0 等值线）/ S / 0-1 材料指示 / 密度 rho。"""
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 6.0), constrained_layout=True)
    PHI = M["phi"]
    ax = axes[0, 0]
    pc = ax.pcolormesh(X, Y, PHI, cmap="coolwarm", shading="auto")
    fig.colorbar(pc, ax=ax)
    if PHI.min() < 0.0 < PHI.max():
        ax.contour(X, Y, PHI, levels=[0.0], colors="k", linewidths=1.2)
    ax.set_title(r"SDF net $\phi$   (black line: $\phi=0$)"
                 + f"\nmin={PHI.min():.4f}  max={PHI.max():.4f}", fontsize=9)

    for ax, F, t, lab in [
        (axes[0, 1], M["S"], r"$S=\frac{1}{2}(1+\tanh(\phi/\epsilon))$, $\epsilon$=%g" % eps,
         "mean"),
        (axes[1, 0], M["mat"], "material indicator (1 = solid, 0 = void)\nblock forced to solid",
         "solid area fraction"),
    ]:
        pc = ax.pcolormesh(X, Y, F, cmap="viridis", vmin=0.0, vmax=1.0, shading="auto")
        fig.colorbar(pc, ax=ax)
        ax.set_title(t + f"\n{lab} = {F.mean():.4f}   min = {F.min():.4f}", fontsize=9)

    ax = axes[1, 1]
    pc = ax.pcolormesh(X, Y, M["rho"], cmap="magma", shading="auto")
    fig.colorbar(pc, ax=ax)
    ax.set_title(r"density $\rho$" + f"\nmin={M['rho'].min():.3f}  max={M['rho'].max():.3f}",
                 fontsize=9)

    for ax in axes.ravel():
        _add_block(ax, block); ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_q(X, Y, EN, KIN, Q, block, om2, path):
    """三联热图：应变能密度 / omega^2 rho|u|^2 / 两者之差 q。"""
    fig, axes = plt.subplots(3, 1, figsize=(7.8, 8.6), constrained_layout=True)
    for ax, F, t in [
        (axes[0], EN, r"$Ae(u){:}e(u)$"),
        (axes[1], KIN, r"$\omega^2\rho|u|^2$   ($\omega^2$=%.6f)" % om2),
        (axes[2], Q, r"$q=Ae(u){:}e(u)-\omega^2\rho|u|^2$"),
    ]:
        _sym_panel(fig, ax, X, Y, F, t, block, clip_pct=99.0)
    fig.savefig(path, dpi=130); plt.close(fig)


def fig_q_surface(X, Y, Q, block, om2, path, clip_pct=99.0):
    """q 的 3D 曲面 + 俯视图（与 viz.plot_q_surface 同款）。

    重块处 rho|u|^2 是别处的几十倍，q 有一个很深的负尖峰，直接画曲面会把其余结构全压平，
    所以曲面按 clip_pct 分位数截断（标题里注明真实范围），俯视图同样用稳健色标。
    """
    vmax = float(np.nanmax(np.abs(Q))) + 1e-30
    v = float(np.nanpercentile(np.abs(Q), clip_pct)) + 1e-30
    clipped = vmax / v > 3.0
    Qc = np.clip(Q, -v, v) if clipped else Q
    fig = plt.figure(figsize=(12.4, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax.plot_surface(X, Y, Qc, cmap="RdBu_r", linewidth=0, antialiased=True)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(r"$q = Ae(u){:}e(u)-\omega^2\rho|u|^2$" + f"   ($\\omega^2$={om2:.6f})"
                 + (f"\nclipped at $\\pm${v:.3g} (true $\\pm${vmax:.3g})" if clipped else ""),
                 fontsize=10)
    fig.colorbar(surf, ax=ax, shrink=0.7)
    ax2 = fig.add_subplot(1, 2, 2)
    _sym_panel(fig, ax2, X, Y, Q, "top view of $q$", block, clip_pct=99.0)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_profiles(X, Y, Q, EN, KIN, block, path):
    """过重块中心的水平剖面（symlog 纵轴：既看得见重块处的深谷，也看得见别处的细节）。"""
    j = int(np.argmin(np.abs(Y[:, 0] - 0.5 * (block[2] + block[3]))))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(X[j], EN[j], lw=1.2, label=r"$Ae(u){:}e(u)$")
    ax.plot(X[j], KIN[j], lw=1.2, label=r"$\omega^2\rho|u|^2$")
    ax.plot(X[j], Q[j], lw=1.6, color="k", label=r"$q$")
    thr = max(1e-8, 0.02 * float(np.abs(Q[j]).max()))
    ax.set_yscale("symlog", linthresh=thr)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvspan(block[0], block[1], color="0.85", zorder=0)
    ax.set_xlabel("x"); ax.set_ylabel("value (symlog)")
    ax.set_title(f"profile at y = {Y[j, 0]:.3f} (through the block)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser(description="从 checkpoint 出图：u / 偏导 / 材料分布 / q")
    ap.add_argument("--ckpt", default=os.path.join("outputs", "model_final.pt"))
    ap.add_argument("--outdir", default=os.path.join("outputs", "figures"))
    ap.add_argument("--nx", type=int, default=641, help="绘图网格 x 方向节点数")
    ap.add_argument("--ny", type=int, default=201, help="绘图网格 y 方向节点数")
    ap.add_argument("--omega2", default="auto",
                    help="omega^2；auto = 从 checkpoint 的 extra 里取 R_final（PINN 自己的值）")
    ap.add_argument("--rho-block", type=float, default=None,
                    help="重块密度（可改量），默认取 cfg.gamma_block")
    ap.add_argument("--rho-solid", type=float, default=1.0, help="有材料区的密度，默认 1")
    ap.add_argument("--rho-mode", choices=["binary", "smooth"], default="binary",
                    help="binary = 0/1 二值（md 里的要求）；smooth = 训练口径的 S_eff")
    ap.add_argument("--block-mode", choices=["sharp", "smooth"], default="sharp",
                    help="重块指示函数：sharp = 硬矩形；smooth = 训练用的 tanh 版")
    ap.add_argument("--mask-void", action="store_true",
                    help="把无材料区的 q 画成空白（当前 SDF 几乎满材料，默认不开）")
    ap.add_argument("--no-normalize", action="store_true",
                    help="不做质量归一化（默认按 int rho|u|^2 = 1 缩放，与训练输出一致）")
    ap.add_argument("--no-check", action="store_true",
                    help="跳过自检（求导对拍 + 求积收敛检查）")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)          # 必须与训练一致
    os.makedirs(args.outdir, exist_ok=True)

    cfg, sdf, nets, extra = core.load_checkpoint(args.ckpt)
    rho_block = cfg.gamma_block if args.rho_block is None else args.rho_block
    if args.omega2 == "auto":
        om2 = float(extra.get("R_final", extra.get("lam_target", extra.get("lam_fem", 0.0))))
        src = "checkpoint 里的 R_final（PINN 的 Rayleigh 商）"
    else:
        om2 = float(args.omega2)
        src = "命令行指定"
    print(f"[载入] {args.ckpt}")
    print(f"[载入] extra = " + ", ".join(f"{k}={float(v):.6g}" for k, v in extra.items()
                                        if not isinstance(v, str)))
    print(f"[参数] omega^2 = {om2:.9f}（{src}）；E={cfg.E:g}, nu={cfg.nu:g}（平面应力）")
    print(f"[参数] rho: 实体={args.rho_solid:g} 空洞=0 重块={rho_block:g}；"
          f"材料={args.rho_mode}，重块指示={args.block_mode}，"
          f"网格 {args.nx}x{args.ny}，质量归一化={'否' if args.no_normalize else '是'}")

    X, Y, XY = make_grid(cfg, args.nx, args.ny)
    shp = X.shape
    g = eval_u_and_grads(nets, XY)
    M = material(sdf, XY, cfg, rho_block, args.rho_solid, args.rho_mode, args.block_mode)
    en, _ = energy_density(g, cfg)

    def N(t):
        return t.squeeze(1).numpy().reshape(shp)

    G = {k: N(v) for k, v in g.items()}
    Mn = {k: N(v) for k, v in M.items()}
    EN = N(en)
    UX, UY = G["ux"], G["uy"]
    rho = Mn["rho"]

    # ---- 质量归一化 + 符号约定：u 是特征向量，整体常数与符号本来就不定 ----
    W = trapz_weights(shp)
    avg = lambda F: float(np.sum(W * F))
    mass = cfg.area * avg(rho * (UX ** 2 + UY ** 2))
    c = 1.0 if args.no_normalize else 1.0 / np.sqrt(mass + 1e-300)
    s = 1.0 if avg(rho * UY) >= 0 else -1.0          # 让 u_y 的质量加权平均为正，出图符号可复现
    a = c * s
    for k in G:
        G[k] = a * G[k]
    UX, UY = G["ux"], G["uy"]
    EN = c * c * EN

    KIN = om2 * rho * (UX ** 2 + UY ** 2)
    Q = EN - KIN

    # ---- 诊断：两种口径下的 Rayleigh 商与 q 的积分 ----
    R_here = avg(EN) / (avg(rho * (UX ** 2 + UY ** 2)) + 1e-300)
    I_q = cfg.area * avg(Q)
    print(f"[校核] 本图口径: R = int A e:e / int rho|u|^2 = {R_here:.9f}"
          f"   相对 omega^2 差 {abs(R_here - om2) / om2:.3e}")
    print(f"[校核] 本图口径: int q dA = {I_q:+.4e}"
          f"   (相对 int A e:e = {I_q / (cfg.area * avg(EN) + 1e-300):+.3e})")

    S_t = Mn["S"]
    chi_s = core.chi_block(XY, cfg.block, cfg.block_smooth_w).squeeze(1).numpy().reshape(shp)
    S_eff = S_t + (1.0 - S_t) * chi_s
    rho_tr = S_eff * (1.0 + (cfg.gamma_block - 1.0) * chi_s)
    MRt = S_eff ** cfg.mask_pow_ray
    R_tr = avg(MRt * EN) / (avg(MRt * rho_tr * (UX ** 2 + UY ** 2)) + 1e-300)
    Iq_tr = cfg.area * avg(MRt * (EN - om2 * rho_tr * (UX ** 2 + UY ** 2)))
    print(f"[校核] 训练口径 (S_eff^{cfg.mask_pow_ray:g} 掩码, gamma={cfg.gamma_block:g}): "
          f"R = {R_tr:.9f}, int S^2 q dA = {Iq_tr:+.4e}")
    print("       —— 训练口径下 int S^2 q dA 应为机器零（omega^2 就是这么定出来的）；"
          "本图口径若换了 rho，积分自然不再为 0。")

    S_bnd = min(S_t[:, 0].min(), S_t[:, -1].min(), S_t[0].min(), S_t[-1].min())
    print(f"[校核] 四边上 S 最小 = {S_bnd:.4f}（sdf_offset={cfg.sdf_offset:g} 时理论值 0.5，"
          f"偏离部分是 SDF 的带内拟合误差）；材料面积占比 = {Mn['mat'].mean():.5f}")

    if not args.no_check:
        errs = check_derivatives(nets, cfg)
        print("[自检] 自动微分 vs 中心差分 相对误差: "
              + "  ".join(f"{k}={v:.1e}" for k, v in errs.items()))
        hx, hy = (args.nx + 1) // 2, (args.ny + 1) // 2
        R_half = rayleigh_train_gauge(nets, sdf, cfg, hx, hy)
        rel = abs(R_half - R_tr) / (abs(R_tr) + 1e-300)
        print(f"[自检] 求积收敛: 训练口径 R 在 {hx}x{hy} 网格 = {R_half:.9f}，"
              f"在 {args.nx}x{args.ny} 网格 = {R_tr:.9f}，相对差 = {rel:.2e}")
        if rel > 1e-3:
            print(f"       警告：两套网格相差 >0.1%，求积还没收敛。原因是 S 在四条边上有一条宽约"
                  f" 4*eps={4 * cfg.eps_heaviside:g} 的过渡带，而当前 dy="
                  f"{cfg.Ly / (args.ny - 1):.4g}；夹持边恰好是应变能最大处，采不准就会整体拉低"
                  f" R。把 --ny 加大直到这个相对差稳定下来，才是 u 真正的 Rayleigh 商。")

    if args.mask_void:
        void = Mn["mat"] < 0.5
        Q = np.where(void, np.nan, Q)
        EN = np.where(void, np.nan, EN)
        KIN = np.where(void, np.nan, KIN)

    p = lambda n: os.path.join(args.outdir, n)
    fig_displacement(X, Y, UX, UY, cfg.block, p("fig1_displacement.png"))
    fig_gradients(X, Y, G, cfg.block, p("fig2_gradients.png"))
    fig_material(X, Y, Mn, cfg.block, cfg.eps_heaviside, p("fig3_material.png"))
    fig_q(X, Y, EN, KIN, Q, cfg.block, om2, p("fig4_q_maps.png"))
    fig_q_surface(X, Y, Q, cfg.block, om2, p("fig5_q_surface.png"))
    fig_profiles(X, Y, Q, EN, KIN, cfg.block, p("fig6_profiles.png"))
    np.savez(p("fields.npz"), X=X, Y=Y, om2=om2, rho=rho, q=Q, en=EN, kin=KIN,
             phi=Mn["phi"], S=Mn["S"], mat=Mn["mat"], **G)
    print(f"[输出] 6 张图 + fields.npz -> {args.outdir}")


if __name__ == "__main__":
    main()
