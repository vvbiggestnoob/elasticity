"""绘图工具（只依赖 numpy / matplotlib / PIL，不依赖 torch，全部输入为 numpy 数组）。

注意：图内文字统一用英文/数学记号，避免用户环境缺 CJK 字体出现方块。
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _add_block(ax, block):
    ax.add_patch(Rectangle((block[0], block[2]), block[1] - block[0], block[3] - block[2],
                           fill=False, edgecolor="k", lw=1.0, ls="--"))


def _sym(ax, fig, X, Y, F, title, block, vmax=None, cmap="RdBu_r"):
    """对称色标的 pcolormesh + colorbar + 重块框，返回 vmax。"""
    if vmax is None:
        vmax = float(np.abs(F).max()) + 1e-30
    pc = ax.pcolormesh(X, Y, F, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(pc, ax=ax)
    _add_block(ax, block)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    return vmax


def align_sign(UX, UY, prev):
    """与上一帧做内积对齐符号，避免动画正负闪烁。返回 (UX, UY, 新的 prev)。"""
    if prev is not None:
        s = float(np.sum(UX * prev[0] + UY * prev[1]))
        if s < 0:
            UX, UY = -UX, -UY
    return UX, UY, (UX.copy(), UY.copy())


def save_snapshot(frame_id, stage, it, X, Y, UX, UY, block, frames_dir):
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8), constrained_layout=True)
    for ax, F, name in zip(axes, (UX, UY), ("u_x", "u_y")):
        vmax = float(np.abs(F).max()) + 1e-30
        pc = ax.pcolormesh(X, Y, F, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        fig.colorbar(pc, ax=ax)
        _add_block(ax, block)
        ax.set_aspect("equal")
        ax.set_title(f"{name}   [{stage}  iter {it}]   max|.|={vmax:.3e}")
    path = os.path.join(frames_dir, f"frame_{frame_id:04d}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_gif(frames_dir, out_path, duration_ms=220):
    """帧合成 GIF。缺 PIL 或帧不足时安静跳过，绝不让训练末尾报错。"""
    try:
        from PIL import Image
    except ImportError:
        print("[viz] 未安装 Pillow，跳过 GIF 合成（帧图仍在 frames/ 下）。")
        return
    files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if len(files) < 2:
        return
    imgs = [Image.open(f).convert("RGB") for f in files]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0)


def plot_fields_pair(X, Y, A, B, titles, block, path, sym=True):
    """上下两幅标量场。sym=True 用红蓝对称色标；False 用 viridis（如 |residual|）。"""
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8), constrained_layout=True)
    for ax, F, t in zip(axes, (A, B), titles):
        if sym:
            vmax = float(np.abs(F).max()) + 1e-30
            pc = ax.pcolormesh(X, Y, F, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        else:
            pc = ax.pcolormesh(X, Y, F, cmap="viridis", shading="auto")
        fig.colorbar(pc, ax=ax)
        _add_block(ax, block)
        ax.set_aspect("equal")
        ax.set_title(t)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_compare_fem(X, Y, uxP, uyP, uxF, uyF, block, path):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 8.0), constrained_layout=True)
    data = [(uxP, "PINN  u_x"), (uyP, "PINN  u_y"),
            (uxF, "FEM  u_x"), (uyF, "FEM  u_y"),
            (uxP - uxF, "diff  u_x"), (uyP - uyF, "diff  u_y")]
    for ax, (F, t) in zip(axes.ravel(), data):
        vmax = float(np.abs(F).max()) + 1e-30
        pc = ax.pcolormesh(X, Y, F, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        fig.colorbar(pc, ax=ax)
        _add_block(ax, block)
        ax.set_aspect("equal")
        ax.set_title(t)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_q_surface(X, Y, Q, block, lam, path,
                   label=r"$q = Ae(u){:}e(u)-\omega^2\rho|u|^2$"):
    """q(x,y) 的 3D 曲面 + 俯视热图。label 用于区分能量式 / 强形式两种 q。"""
    fig = plt.figure(figsize=(12.0, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax.plot_surface(X, Y, Q, cmap="RdBu_r", linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(label + f"   ($\\omega^2$={lam:.6f})")
    fig.colorbar(surf, ax=ax, shrink=0.7)
    ax2 = fig.add_subplot(1, 2, 2)
    vmax = float(np.abs(Q).max()) + 1e-30
    # 重块处 rho|u|^2 极大，q 有一个很深的负尖峰，会把其余结构全压平；
    # 俯视图用 99 分位数做稳健色标，标题里注明被截断。
    vrob = float(np.percentile(np.abs(Q), 99)) + 1e-30
    clip = vmax / vrob > 3.0
    v = vrob if clip else vmax
    pc = ax2.pcolormesh(X, Y, Q, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    fig.colorbar(pc, ax=ax2, extend="both" if clip else "neither")
    _add_block(ax2, block)
    ax2.set_aspect("equal")
    ax2.set_title("top view of q" + (f"   (colour clipped at $\\pm${v:.3g},"
                                     f" true range $\\pm${vmax:.3g})" if clip else ""),
                  fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_loss_curves(hist, lam_fem, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    it = np.array(hist["iter"])
    for k in ("pde", "cons", "dir", "neu", "norm", "ray", "pde_blk", "cons_blk"):
        if k in hist and len(hist[k]) == len(it):
            axes[0].semilogy(it, np.maximum(np.array(hist[k]), 1e-18), label=k, lw=1.2)
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].set_xlabel("iter")
    axes[0].set_title("loss terms")
    axes[1].plot(it, np.array(hist["R"]), label="Rayleigh R(u)")
    axes[1].axhline(lam_fem, color="k", ls="--", label=r"$\omega^2_{FEM}$")
    axes[1].set_xlabel("iter")
    axes[1].set_title("Rayleigh quotient")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ====================== 新增 1：SDF 预训练 / 材料场诊断 ======================

def plot_sdf_material(X, Y, PHI, PHI_T, GN, S, S_EFF, RHO, block, path,
                      eps=None, init=""):
    """SDF 预训练结果的六联图。

    上排 phi_net / phi_target / |grad phi|，下排 S / S_eff / rho。

    读图要点：
    - phi 面板画了黑色的 phi=0 等值线（材料/空洞界面）；常数初始化时域内没有这条线；
    - phi 面板色标是自适应的，全域几乎常数时看到的花纹只是数值抖动，看标题的 std；
    - |grad phi| 是"是不是距离函数"的判据：距离函数应 ≈ 1（中轴线、孔边会被网络抹平
      而偏小），常数初始化则 ≈ 0；
    - S 与 S_eff 固定色标 [0,1]，满材料时整片贴着 1。若边界处只有 0.5，说明零水平集
      压在域边界上，夹持边会变成"半材料"；
    - rho 面板应当只在重块处出现 gamma 倍的方块。
    """
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 6.4), constrained_layout=True)
    ttl = r"SDF net output $\phi$" + (f"   [init={init}]" if init else "")
    ttl += f"\nmin={PHI.min():.4f}  max={PHI.max():.4f}  std={PHI.std():.2e}"
    err = np.abs(PHI - PHI_T)
    panels = [
        (PHI, ttl, "coolwarm", None, True),
        (PHI_T, (r"target $\phi$ (analytic)"
                 + f"\nfit error: max={err.max():.2e}  RMS={np.sqrt((err**2).mean()):.2e}"),
         "coolwarm", None, True),
        (GN, (r"$|\nabla\phi|$  (distance function $\Rightarrow$ 1, const $\Rightarrow$ 0)"
              + f"\nmin={GN.min():.3f}  mean={GN.mean():.3f}  max={GN.max():.3f}"),
         "viridis", None, False),
        (S, (r"$S=\frac{1}{2}(1+\tanh(\phi/\epsilon))$"
             + (rf",  $\epsilon$={eps:g}" if eps is not None else "")
             + f"\nmin={S.min():.6f}  max={S.max():.6f}"), "viridis", (0.0, 1.0), False),
        (S_EFF, (r"$S_{eff}=S+(1-S)\chi_{block}$ (block forced to 1)"
                 + f"\nmin={S_EFF.min():.6f}  max={S_EFF.max():.6f}"),
         "viridis", (0.0, 1.0), False),
        (RHO, (r"$\rho=S_{eff}\,[1+(\gamma-1)\chi_{block}]$"
               + f"\nmin={RHO.min():.4f}  max={RHO.max():.4f}"), "magma", None, False),
    ]
    for ax, (F, t, cm, rng, zero_line) in zip(axes.ravel(), panels):
        kw = dict(cmap=cm, shading="auto")
        if rng is not None:
            kw.update(vmin=rng[0], vmax=rng[1])
        pc = ax.pcolormesh(X, Y, F, **kw)
        fig.colorbar(pc, ax=ax)
        if zero_line and F.min() < 0.0 < F.max():
            ax.contour(X, Y, F, levels=[0.0], colors="k", linewidths=1.2)
        _add_block(ax, block)
        ax.set_aspect("equal")
        ax.set_title(t, fontsize=9)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def plot_sdf_pretrain(hist, target, xprof, rho_prof, S_prof, path):
    """左：预训练 MSE 收敛曲线；右：过重块中心的一条水平线上 rho 与 S 的剖面。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    h = np.maximum(np.asarray(hist, dtype=float), 1e-20)
    axes[0].semilogy(np.arange(1, len(h) + 1), h, lw=1.0)
    axes[0].set_xlabel("pretrain iter")
    axes[0].set_ylabel(r"MSE$_{\rm int}+w_{\rm bnd}\cdot$MSE$_{\rm bnd}$  $(\phi-\phi_{target})$")
    axes[0].set_title(f"SDF pretraining (target $\\phi$={target:g}), final={h[-1]:.2e}")
    axes[0].grid(alpha=0.3)

    ax = axes[1]
    ax.plot(xprof, rho_prof, lw=1.4, label=r"$\rho$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$")
    ax2 = ax.twinx()
    ax2.plot(xprof, S_prof, lw=1.0, color="tab:orange", ls="--", label=r"$S_{eff}$")
    ax2.set_ylabel(r"$S_{eff}$")
    ax2.set_ylim(-0.05, 1.15)
    ax.set_title("profile through the block centre line")
    ln1, lb1 = ax.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(ln1 + ln2, lb1 + lb2, loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============ 新增 2：sigma 网络（一阶导） vs u 网络（二阶导） 对比 ============

def plot_stress_compare(X, Y, S_nn, S_u, block, path):
    """3x3：每行一个应力分量，列 = sigma^NN / sigma(u)=Ae(u) / 差。

    S_nn, S_u 均为 (sxx, syy, sxy) 的三元组。前两列共用色标，便于直接比大小。
    """
    names = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\sigma_{xy}$"]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.2), constrained_layout=True)
    for r in range(3):
        A, B = S_nn[r], S_u[r]
        v = max(float(np.abs(A).max()), float(np.abs(B).max())) + 1e-30
        _sym(axes[r, 0], fig, X, Y, A, f"{names[r]} from $\\sigma$-net", block, v)
        _sym(axes[r, 1], fig, X, Y, B, f"{names[r]} from $Ae(u)$ (1st deriv of u)", block, v)
        d = A - B
        rel = float(np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(B ** 2)) + 1e-30))
        _sym(axes[r, 2], fig, X, Y, d, f"difference   rel.RMS={rel:.2e}", block)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def plot_div_compare(X, Y, Ds, Du, block, path):
    """2x3：行 = x / y 分量，列 = div(sigma^NN) / div(Ae(u)) / 差。

    Ds = (div1_s, div2_s) 来自 sigma 网络的一阶导；
    Du = (div1_u, div2_u) 来自 u 网络的二阶导（Navier 算子）。
    理想收敛时两者应当处处重合，差场的量级就是混合格式的自洽性误差。
    """
    comp = ["x", "y"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.8), constrained_layout=True)
    for r in range(2):
        A, B = Ds[r], Du[r]
        v = max(float(np.abs(A).max()), float(np.abs(B).max())) + 1e-30
        _sym(axes[r, 0], fig, X, Y, A,
             r"$(\nabla\!\cdot\!\sigma^{NN})_%s$   (1st deriv of $\sigma$-net)" % comp[r],
             block, v)
        _sym(axes[r, 1], fig, X, Y, B,
             r"$(\nabla\!\cdot\!Ae(u))_%s$   (2nd deriv of $u$-net)" % comp[r],
             block, v)
        d = A - B
        rel = float(np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(B ** 2)) + 1e-30))
        _sym(axes[r, 2], fig, X, Y, d, f"difference   rel.RMS={rel:.2e}", block)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def plot_residual_compare(X, Y, Rs, Ru, block, path):
    """2x2：|r| = |div sigma + omega^2 rho u|，左列 sigma 网络路线，右列 u 二阶导路线。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.8), constrained_layout=True)
    data = [(np.abs(Rs[0]), r"$|r_x|$ from $\sigma$-net"),
            (np.abs(Ru[0]), r"$|r_x|$ from $Ae(u)$ (2nd deriv)"),
            (np.abs(Rs[1]), r"$|r_y|$ from $\sigma$-net"),
            (np.abs(Ru[1]), r"$|r_y|$ from $Ae(u)$ (2nd deriv)")]
    for ax, (F, t) in zip(axes.ravel(), data):
        pc = ax.pcolormesh(X, Y, F, cmap="viridis", shading="auto")
        fig.colorbar(pc, ax=ax)
        _add_block(ax, block)
        ax.set_aspect("equal")
        ax.set_title(t + f"   RMS={np.sqrt(np.mean(F ** 2)):.2e}", fontsize=9)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def plot_lbfgs_curve(blocks, train_loss, val_loss, path):
    """L-BFGS 每一块结束时的训练损失 vs 独立验证点集损失：判断该不该再多跑几块。"""
    if len(blocks) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.semilogy(blocks, np.maximum(train_loss, 1e-20), "o-", ms=3, label="train (fixed points)")
    ax.semilogy(blocks, np.maximum(val_loss, 1e-20), "s--", ms=3, label="validation (unseen points)")
    k = int(np.argmin(val_loss))
    ax.axvline(blocks[k], color="k", ls=":", lw=1.0)
    ax.set_xlabel("L-BFGS block")
    ax.set_ylabel("total loss")
    ax.set_title(f"L-BFGS: best validation at block {blocks[k]}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_q_compare(X, Y, Qp, Qf, block, lam_p, lam_f, path, jc=None):
    """PINN 与 FEM 的 q 对照。

    上排：两张俯视图，共用同一套稳健色标（99 分位），可直接比强弱；
    下排：差场（自带相对 RMS）+ 过重块中心线的剖面（symlog 纵轴，
          既看得见重块处的深谷，也看得见其余位置的细节）。
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 6.6), constrained_layout=True)
    v = max(float(np.percentile(np.abs(Qp), 99)),
            float(np.percentile(np.abs(Qf), 99))) + 1e-30
    vt = max(float(np.abs(Qp).max()), float(np.abs(Qf).max()))
    clip = vt / v > 3.0
    for ax, F, t in [(axes[0, 0], Qp, f"PINN  $q$   ($\\omega^2$={lam_p:.6f})"),
                     (axes[0, 1], Qf, f"FEM   $q$   ($\\omega^2$={lam_f:.6f})")]:
        pc = ax.pcolormesh(X, Y, F, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
        fig.colorbar(pc, ax=ax, extend="both" if clip else "neither")
        _add_block(ax, block)
        ax.set_aspect("equal")
        ax.set_title(t + (f"   (colour clipped at $\\pm${v:.3g})" if clip else ""),
                     fontsize=9)

    d = Qp - Qf
    rel = float(np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(Qf ** 2)) + 1e-30))
    _sym(axes[1, 0], fig, X, Y, d, f"PINN $-$ FEM    rel.RMS={rel:.2e}", block)

    ax = axes[1, 1]
    if jc is None:
        jc = X.shape[0] // 2
    ax.plot(X[jc], Qp[jc], lw=1.5, label="PINN")
    ax.plot(X[jc], Qf[jc], lw=1.2, ls="--", label="FEM")
    thr = max(1e-6, 0.02 * float(np.abs(Qf[jc]).max()))
    ax.set_yscale("symlog", linthresh=thr)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("q  (symlog)")
    ax.set_title(f"profile at y={Y[jc, 0]:.3f} (through the block)", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=130)
    plt.close(fig)
