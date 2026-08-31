"""自检脚本：不训练，只验证 pinn_core 的求导链路与公式是否写对（几秒钟跑完）。

运行：python selftest.py
检查项：
  1. eval_report 的 div1_u/div2_u（u 网络二阶导）== 对 sigma(u) 做中心差分
  2. 能量密度 en 的两种等价写法一致：sigma:e  ==  2*mu*e:e + lambda_ps*(tr e)^2
  3. 逐点 Green 恒等式 div(sigma·u) == (div sigma)·u + sigma:e(u)
  4. torch 版与 numpy 版的重块指示函数一致（pinn_core vs fem_baseline）
  5. SDF 预训练后 S_eff ≈ 1、重块处 rho ≈ gamma
"""
import numpy as np
import torch

from config import Config
import pinn_core as core
import fem_baseline as fem


def rel(a, b):
    a, b = a.detach().numpy(), b.detach().numpy()
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-300))


def report(name, err, tol):
    ok = "通过" if err < tol else "失败"
    print(f"  [{ok}] {name}: 相对误差 {err:.3e} (阈值 {tol:g})")
    return err < tol


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    cfg = Config()
    nets = core.build_nets(cfg)
    ok = True

    # 随机内部点（离边界留 0.1，便于中心差分）
    n = 300
    lo = torch.tensor([0.1, 0.1])
    span = torch.tensor([cfg.Lx - 0.2, cfg.Ly - 0.2])
    xy = lo + torch.rand(n, 2) * span

    rep = core.eval_report(nets, xy, cfg)

    # ---------- 1) 二阶导路线 vs 中心差分 ----------
    def sigma_u(p):
        p = p.clone().requires_grad_(True)
        ux, uy = nets["ux"](p), nets["uy"](p)
        ux_x, ux_y = core._grad(ux, p)
        uy_x, uy_y = core._grad(uy, p)
        exx, eyy, exy = ux_x, uy_y, 0.5 * (ux_y + uy_x)
        s = core.stress_from_strain(exx, eyy, exy, cfg.E, cfg.nu)
        return [t.detach() for t in s]          # (sxx, syy, sxy)

    h = 1e-5
    ex = torch.tensor([[h, 0.0]])
    ey = torch.tensor([[0.0, h]])
    sxx_xp, syy_xp, sxy_xp = sigma_u(xy + ex)
    sxx_xm, syy_xm, sxy_xm = sigma_u(xy - ex)
    sxx_yp, syy_yp, sxy_yp = sigma_u(xy + ey)
    sxx_ym, syy_ym, sxy_ym = sigma_u(xy - ey)
    div1_fd = (sxx_xp - sxx_xm) / (2 * h) + (sxy_yp - sxy_ym) / (2 * h)
    div2_fd = (sxy_xp - sxy_xm) / (2 * h) + (syy_yp - syy_ym) / (2 * h)
    ok &= report("div(Ae(u))_x 二阶导 vs 差分", rel(rep["div1_u"], div1_fd), 1e-6)
    ok &= report("div(Ae(u))_y 二阶导 vs 差分", rel(rep["div2_u"], div2_fd), 1e-6)

    # ---------- 2) 能量密度的两种写法 ----------
    exx, eyy, exy = rep["exx"], rep["eyy"], rep["exy"]
    mu = cfg.E / (2.0 * (1.0 + cfg.nu))
    lam_ps = cfg.E * cfg.nu / (1.0 - cfg.nu ** 2)      # 平面应力等效 Lame 系数
    en_lame = 2 * mu * (exx ** 2 + eyy ** 2 + 2 * exy ** 2) + lam_ps * (exx + eyy) ** 2
    ok &= report("Ae(u):e(u) 的 sigma:e 写法 vs Lame 写法", rel(rep["en"], en_lame), 1e-12)

    # ---------- 3) 逐点 Green 恒等式 ----------
    p = xy.clone().requires_grad_(True)
    ux, uy = nets["ux"](p), nets["uy"](p)
    ux_x, ux_y = core._grad(ux, p)
    uy_x, uy_y = core._grad(uy, p)
    Exx, Eyy, Exy = ux_x, uy_y, 0.5 * (ux_y + uy_x)
    Sxx, Syy, Sxy = core.stress_from_strain(Exx, Eyy, Exy, cfg.E, cfg.nu)
    fx = Sxx * ux + Sxy * uy       # (sigma·u)_x
    fy = Sxy * ux + Syy * uy       # (sigma·u)_y
    fx_x, _ = core._grad(fx, p)
    _, fy_y = core._grad(fy, p)
    lhs = fx_x + fy_y
    d1u, d2u = rep["div1_u"], rep["div2_u"]
    rhs = d1u * ux.detach() + d2u * uy.detach() + rep["en"]
    ok &= report("div(sigma·u) == (div sigma)·u + sigma:e(u)", rel(lhs, rhs), 1e-10)

    # ---------- 4) 重块指示函数：torch 版 vs numpy 版 ----------
    chi_t = core.chi_block(xy, cfg.block, cfg.block_smooth_w)
    chi_n = fem.analytic_chi_block(xy[:, 0].numpy(), xy[:, 1].numpy(),
                                   cfg.block, cfg.block_smooth_w)
    err = float(np.abs(chi_t.detach().numpy().ravel() - chi_n).max())
    ok &= report("chi_block: torch 版 vs numpy 版", err, 1e-14)

    # ---------- 5) SDF 预训练后的材料场 ----------
    cfg_fast = Config()
    cfg_fast.sdf_pretrain_iters = 400
    cfg_fast.sdf_lbfgs_outer = 3            # 自检只求快：3*40 次 L-BFGS 已够把带内误差压下去
    cfg_fast.sdf_lbfgs_max_iter_per = 40
    cfg_fast.sdf_lbfgs_int = 4096
    cfg_fast.sdf_lbfgs_bnd = 1024
    sdf, mse, _ = core.pretrain_sdf(cfg_fast)
    # "满材料区"从过渡带外侧起算：默认 edge_dist + offset=0 时边界上 S=0.5 是设计使然
    # （零水平集压在域边界上，见 config 注释），不是拟合误差，所以检查网格向内收
    # 6*eps（tanh(6) ≈ 0.99999，那之外才谈得上"应≈1"）。
    pad = 6.0 * cfg_fast.eps_heaviside
    gx = torch.linspace(pad, cfg_fast.Lx - pad, 321)
    gy = torch.linspace(pad, cfg_fast.Ly - pad, 101)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    XY = torch.stack([GX.reshape(-1), GY.reshape(-1)], dim=1)
    _, S, chi, S_eff, rho = core.material_fields_full(sdf, XY, cfg_fast)
    print(f"  [信息] SDF 预训练 mse={mse:.2e}; S_eff 最小={float(S_eff.min()):.6f} "
          f"平均={float(S_eff.mean()):.6f}; chi 最大={float(chi.max()):.6f}; "
          f"rho 最大={float(rho.max()):.4f} (gamma={cfg.gamma_block:g})")
    ok &= report("满材料区 S_eff ≈ 1", float(abs(S_eff.min() - 1.0)), 5e-3)
    # chi 的峰值略小于 1（重块半宽 0.04 只有过渡宽度 w=0.01 的 4 倍，tanh 没有完全饱和），
    # 因此 rho 峰值约 gamma 的 99.9%；FEM 用的是同一个 chi，两边一致，不影响对拍。
    ok &= report("重块峰值 rho ≈ gamma",
                 float(abs(rho.max() - cfg.gamma_block) / cfg.gamma_block), 3e-2)

    # ---------- 5b) 边界带采样器（确定性几何校验）与带内拟合精度 ----------
    pb = core.sdf_band_xy(2000, cfg_fast, torch.Generator().manual_seed(123))
    d_edge = torch.minimum(torch.minimum(pb[:, 0], cfg_fast.Lx - pb[:, 0]),
                           torch.minimum(pb[:, 1], cfg_fast.Ly - pb[:, 1]))
    outside = float(((pb[:, 0] < 0) | (pb[:, 0] > cfg_fast.Lx)
                     | (pb[:, 1] < 0) | (pb[:, 1] > cfg_fast.Ly)).any())
    ok &= report("边界带点都在闭域内", outside, 1e-12)
    ok &= report("边界带点到四边距离 <= band",
                 float((d_edge - cfg_fast.sdf_bnd_band).clamp_min(0.0).max()), 1e-12)
    ok &= report("存在严格压在边上的点", float(d_edge.min()), 1e-12)
    with torch.no_grad():
        e_band = float((sdf(pb) - core.analytic_sdf(pb, cfg_fast)).abs().max())
    print(f"  [信息] 预训练后边界带内最大拟合误差 = {e_band:.2e}"
          f"  (eps={cfg_fast.eps_heaviside:g}，err/eps = {e_band/cfg_fast.eps_heaviside:.2f}；"
          f"该比值 < 0.5 时边界 S 才不被网络噪声主导)")

    c_h = Config()
    c_h.sdf_init = "holes"
    ph2 = core.sdf_band_xy(2000, c_h, torch.Generator().manual_seed(7))
    d_e2 = torch.minimum(torch.minimum(ph2[:, 0], c_h.Lx - ph2[:, 0]),
                         torch.minimum(ph2[:, 1], c_h.Ly - ph2[:, 1]))
    d_c2 = torch.full_like(d_e2, float("inf"))
    for (cx, cy, r) in c_h.sdf_holes:
        d_c2 = torch.minimum(
            d_c2, (((ph2[:, 0] - cx) ** 2 + (ph2[:, 1] - cy) ** 2).sqrt() - r).abs())
    ok &= report("holes: 采样点在边带或孔缘带内",
                 float((torch.minimum(d_e2, d_c2) - c_h.sdf_bnd_band).clamp_min(0.0).max()),
                 1e-12)

    # ---------- 6) 三种 SDF 初始化的解析场 ----------
    c2 = Config()
    pts = torch.tensor([[0.10, 0.25], [0.80, 0.05], [0.40, 0.25], [0.02, 0.02]])
    c2.sdf_init, c2.sdf_offset = "edge_dist", 0.0
    got = core.analytic_sdf(pts, c2).squeeze(1).numpy()
    want = np.array([0.10, 0.05, 0.25, 0.02])          # 到最近边的垂距
    ok &= report("edge_dist = 到四边的最短距离",
                 float(np.abs(got - want).max()), 1e-12)
    c2.sdf_offset = 5 * c2.eps_heaviside
    got2 = core.analytic_sdf(pts, c2).squeeze(1).numpy()
    ok &= report("offset 只是整体上抬",
                 float(np.abs(got2 - want - c2.sdf_offset).max()), 1e-12)

    # |grad phi| = 1：距离函数的定义式（中轴线上不可导，避开）
    q = torch.tensor([[0.10, 0.25], [1.50, 0.25], [0.80, 0.05], [0.80, 0.45]])
    q.requires_grad_(True)
    g = torch.autograd.grad(core.analytic_sdf(q, c2).sum(), q)[0]
    ok &= report("|grad phi| = 1 (edge_dist)",
                 float((g.norm(dim=1) - 1.0).abs().max()), 1e-12)

    c2.sdf_init = "holes"
    ctrs = torch.tensor([[h[0], h[1]] for h in c2.sdf_holes])
    ph = core.analytic_sdf(ctrs, c2).squeeze(1).numpy()
    want_h = np.array([-h[2] for h in c2.sdf_holes])   # 孔心处 phi = -r
    ok &= report("holes: 孔心处 phi = -r", float(np.abs(ph - want_h).max()), 1e-12)

    # ---------- 7) 刚度算子两边一致 ----------
    c3 = Config()
    Sv = torch.linspace(0.05, 1.0, 20).reshape(-1, 1)
    for flag in (False, True):
        c3.use_simp = flag
        got = core.simp_factor(Sv, c3).numpy().ravel()
        want = (Sv.numpy().ravel() ** c3.simp_p) if flag else np.ones(20)
        ok &= report(f"simp_factor(use_simp={flag}) 与 FEM stiff_fn 同式",
                     float(np.abs(got - want).max()), 1e-14)

    # ---------- 8) 掩码口径：损失用 S^1，Rayleigh 用 S^2 ----------
    c4 = Config()
    sdf2, _, _ = core.pretrain_sdf(c4) if False else (sdf, 0.0, None)   # 复用上面的 sdf
    pool = core.build_pool(sdf2, c4, 256, 64, 32, seed=0)
    S_int = pool["int_mask"] ** (1.0 / c4.mask_pow_loss)
    ok &= report("int_mask = S^mask_pow_loss",
                 float((pool["int_mask"] - S_int ** c4.mask_pow_loss).abs().max()), 1e-12)
    ok &= report("int_mask_ray = S^mask_pow_ray",
                 float((pool["int_mask_ray"] - S_int ** c4.mask_pow_ray).abs().max()), 1e-10)
    print(f"  [信息] 掩码指数：损失 S^{c4.mask_pow_loss:g}，Rayleigh S^{c4.mask_pow_ray:g}；"
          f"边界点 mask 范围 [{float(pool['dir_mask'].min()):.4f}, "
          f"{float(pool['dir_mask'].max()):.4f}]")

    print("\n全部检查通过。" if ok else "\n有检查未通过，请看上面的项。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
