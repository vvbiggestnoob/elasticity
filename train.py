"""主脚本：SDF 初始化 -> FEM 基准 omega^2 -> 5 网络混合格式 PINN 两阶段训练 -> 评估出图。

运行：python train.py        （所有开关见 config.py，输出在 outputs/）
流程严格遵循：先初始化 SDF 得到 rho（本阶段冻结），FEM 在该 rho 上求 omega^2 作为锚点，
然后无监督训练方程（FEM 不做监督，仅提供标量 omega^2 与事后对拍验证）。

输出清单（outputs/）：
    sdf_material.png        SDF 预训练结果：phi / S / S_eff / rho 四联图
    sdf_pretrain.png        SDF 预训练收敛曲线 + 过重块中心线的 rho、S 剖面
    fem_mode.png            FEM 参考模态
    frames/ + evolution.gif ux、uy 的训练演变
    loss_curves.png         各损失项与 Rayleigh 商
    lbfgs_curve.png         L-BFGS 每块的训练损失 vs 独立验证损失（判断该不该多跑）
    final_fields.png        最终位移场
    compare_fem.png         PINN vs FEM 模态对拍
    residual_maps.png       |r| （sigma 网络路线）
    residual_compare.png    两条路线的残差对比
    stress_compare.png      sigma^NN  vs  Ae(u)      （sigma 一阶导 vs u 一阶导）
    div_compare.png         div sigma^NN vs div Ae(u)（sigma 一阶导 vs u 二阶导）
    q_surface_3d.png        q = Ae(u):e(u) - omega^2 rho|u|^2   （能量式，只含 u 的一阶导）
    q_surface_strong_3d.png q~ = -(div Ae(u))·u - omega^2 rho|u|^2（强形式，含 u 的二阶导）
    q_surface_fem_3d.png    FEM 版 q（把 FEM 模态代回 Q4 单元算，与 K、M 同一套形函数）
    q_compare.png           PINN q vs FEM q：俯视图对照 + 差场 + 过重块中心线剖面
    model_sdf.pt / model_adam.pt / model_final.pt   权重（见 pinn_core.load_checkpoint）
    history.npz / final_fields.npz / summary.txt
"""
import os
import time
import copy
import numpy as np
import torch

from config import Config
import fem_baseline as fem
import pinn_core as core
import viz


# ---------------------------------------------------------------- FEM 参考

def fem_reference(cfg, sdf):
    """用冻结的 SDF 生成单元密度/刚度场，跑粗、细两套网格的 FEM。"""

    def fields_np(xc, yc):
        xy = torch.tensor(np.stack([xc, yc], axis=1))
        S, rho = core.material_fields(sdf, xy, cfg)
        return S.squeeze(1).numpy(), rho.squeeze(1).numpy()

    def dens_fn(xc, yc):
        return fields_np(xc, yc)[1]

    def stiff_fn(xc, yc):
        S = fields_np(xc, yc)[0]
        # 与 pinn_core.simp_factor 同一个式子：两边必须是同一个算子
        return S ** cfg.simp_p if cfg.use_simp else np.ones_like(S)

    res_c = fem.solve_plane_stress_eig(cfg.nx_fem_coarse, cfg.ny_fem_coarse,
                                       cfg.Lx, cfg.Ly, cfg.E, cfg.nu,
                                       dens_fn, stiff_fn, cfg.n_modes_fem)
    res_f = fem.solve_plane_stress_eig(cfg.nx_fem, cfg.ny_fem,
                                       cfg.Lx, cfg.Ly, cfg.E, cfg.nu,
                                       dens_fn, stiff_fn, cfg.n_modes_fem)
    return res_c, res_f, dens_fn, stiff_fn


# ---------------------------------------------------------------- SDF 诊断

def sdf_diagnostics(cfg, sdf, sdf_hist):
    """SDF 预训练完的自检：打印统计量 + 画 phi / S / S_eff / rho 与剖面。"""
    xs = np.linspace(0.0, cfg.Lx, cfg.sdf_plot_nx)
    ys = np.linspace(0.0, cfg.Ly, cfg.sdf_plot_ny)
    Xp, Yp = np.meshgrid(xs, ys)
    XYp = torch.tensor(np.stack([Xp.ravel(), Yp.ravel()], axis=1))
    phi, S, chi, S_eff, rho = core.material_fields_full(sdf, XYp, cfg)
    with torch.no_grad():
        phi_t = core.analytic_sdf(XYp, cfg)
    gn = core.sdf_grad_norm(sdf, XYp)          # |grad phi|：是不是距离函数看这个

    def R(t):
        return t.squeeze(1).numpy().reshape(Xp.shape)

    PHI, PHI_T, GN = R(phi), R(phi_t), R(gn)
    Sg, SE, RHO = R(S), R(S_eff), R(rho)
    viz.plot_sdf_material(Xp, Yp, PHI, PHI_T, GN, Sg, SE, RHO, cfg.block,
                          os.path.join(cfg.outdir, "sdf_material.png"),
                          eps=cfg.eps_heaviside, init=cfg.sdf_init)

    jc = int(np.argmin(np.abs(ys - 0.5 * (cfg.block[2] + cfg.block[3]))))
    viz.plot_sdf_pretrain(sdf_hist, cfg.sdf_target, xs, RHO[jc], SE[jc],
                          os.path.join(cfg.outdir, "sdf_pretrain.png"))

    fit_err = float(np.abs(PHI - PHI_T).max())
    # 要害在边界带：S=H(phi/eps) 只对 |phi| <~ 4*eps 的窄带敏感，域四边/孔缘正落在带里；
    # 全域最大误差通常出在中轴线折点上，那里 |phi| >> eps、S 已饱和，误差无害。
    band = cfg.sdf_bnd_band
    in_band = np.minimum(np.minimum(Xp, cfg.Lx - Xp),
                         np.minimum(Yp, cfg.Ly - Yp)) <= band
    if cfg.sdf_init == "holes":
        for (cx, cy, r) in cfg.sdf_holes:
            in_band |= np.abs(np.hypot(Xp - cx, Yp - cy) - r) <= band
    band_err = float(np.abs(PHI - PHI_T)[in_band].max())
    S_T = 0.5 * (1.0 + np.tanh(PHI_T / cfg.eps_heaviside))
    S_dist = np.abs(Sg - S_T)
    area_frac = float(np.mean(SE))
    S_bnd = float(min(Sg[:, 0].min(), Sg[:, -1].min(), Sg[0].min(), Sg[-1].min()))
    print(f"[SDF] 初始化方式 = {cfg.sdf_init}"
          + (f"  (常数 {cfg.sdf_target:g})" if cfg.sdf_init == "const"
             else f"  (offset={cfg.sdf_offset:g} = {cfg.sdf_offset/cfg.eps_heaviside:.1f}*eps)"))
    print(f"[SDF] phi: min={PHI.min():.4f} max={PHI.max():.4f} std={PHI.std():.2e}"
          f"  拟合误差 max 全域={fit_err:.2e} 边界带内={band_err:.2e}"
          f" (band_err/eps={band_err/cfg.eps_heaviside:.2f})")
    print(f"[SDF] S 失真 |S - S_target|: 带内 max={float(S_dist[in_band].max()):.3e}"
          f"  全域均值={float(S_dist.mean()):.3e}")
    if band_err > cfg.eps_heaviside:
        print(f"[SDF] 警告：边界带拟合误差已超过 eps={cfg.eps_heaviside:g}，S 会被明显扭曲，"
              f"请加大 sdf_lbfgs_outer、调高 sdf_w_bnd 或把 sdf_pretrain_iters 调大。")
    print(f"[SDF] |grad phi|: min={GN.min():.3f} 平均={GN.mean():.3f} max={GN.max():.3f}"
          f"  (距离函数应≈1，常数初始化≈0)")
    print(f"[SDF] S_eff: min={SE.min():.6f} max={SE.max():.6f} 平均={area_frac:.6f}"
          f"  (满材料应≈1)；四边上 S 最小={S_bnd:.4f}")
    if S_bnd < 0.9:
        band = 4.0 * cfg.eps_heaviside
        print(f"[SDF] 提示：域边界上 S={S_bnd:.3f}（零水平集压在边界上，这是 sdf_offset=0 的必然结果，"
              f"与 eps 无关）。")
        print(f"       - 边界损失被整体乘了 {S_bnd ** cfg.mask_pow_loss:.3f}，"
              f"等价于把 w_dir/w_neu 打了折；想抵消就把这两个权重除以它。")
        print(f"       - 过渡带宽约 4*eps={band:.4f}，占板厚 {band/cfg.Ly:.1%}，"
              f"细网格单元高 {cfg.Ly/cfg.ny_fem:g}、粗网格 {cfg.Ly/cfg.ny_fem_coarse:g}；"
              f"eps 再小就低于网格分辨率了。")
        if band_err > 0.5 * cfg.eps_heaviside:
            print(f"       - 警告：边界带拟合误差 {band_err:.2e} 已达 eps 的一半以上，"
                  f"边界附近的 S 会由网络噪声主导，请加大 sdf_lbfgs_outer / sdf_w_bnd，"
                  f"或放宽 eps。")
    print(f"[SDF] rho: min={RHO.min():.4f} max={RHO.max():.4f}"
          f"  (重块处应≈{cfg.gamma_block:g})")


# ---------------------------------------------------------------- 主流程

def main(cfg=None):
    cfg = cfg or Config()
    torch.set_default_dtype(torch.float64)          # 全程 float64
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)
    frames_dir = os.path.join(cfg.outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    t_start = time.time()

    # ---------- 1) SDF 初始化并冻结（全域满材料 S≈1，重块由指示函数强制） ----------
    t0 = time.time()
    sdf, sdf_mse, sdf_hist = core.pretrain_sdf(cfg)
    print(f"[SDF] 预训练完成（Adam + 边界带加权 + L-BFGS 微调）"
          f"组合损失={sdf_mse:.3e}  ({time.time()-t0:.1f}s)")
    sdf_diagnostics(cfg, sdf, sdf_hist)
    if cfg.save_ckpt:
        # 只存 SDF（此时 5 个物理场网络还没建）；完整权重见 model_adam.pt / model_final.pt
        torch.save(dict(sdf=sdf.state_dict(), config=dict(cfg.__dict__)),
                   os.path.join(cfg.outdir, "model_sdf.pt"))

    # ---------- 2) FEM 基准：在初始化后的 rho 上求 omega^2 ----------
    t0 = time.time()
    res_c, res_f, dens_fn, stiff_fn = fem_reference(cfg, sdf)
    lam_f = float(res_f["lams"][cfg.mode_index])
    lam_c = float(res_c["lams"][cfg.mode_index])
    lam_rich = float(fem.richardson(lam_c, lam_f))
    mesh_rel = abs(lam_c - lam_f) / lam_f
    lam_tgt = lam_rich if cfg.use_richardson_target else lam_f   # 训练锚点
    print(f"[FEM] 粗网格 omega^2 = {np.array2string(res_c['lams'], precision=6)}")
    print(f"[FEM] 细网格 omega^2 = {np.array2string(res_f['lams'], precision=6)}")
    print(f"[FEM] 目标模态 index={cfg.mode_index}: omega^2_FEM={lam_f:.6f}"
          f"  粗细网格相对差={mesh_rel:.2e}  ({time.time()-t0:.1f}s)")
    print(f"[FEM] Richardson 外推 = {lam_rich:.6f}"
          f"  (Q4 特征值从上方收敛，这是更接近连续解的估计)")
    print(f"[FEM] 本次训练锚点 omega^2_target = {lam_tgt:.6f}"
          f"  ({'Richardson' if cfg.use_richardson_target else '细网格 FEM'})")

    xs, ys = res_f["xs"], res_f["ys"]
    X, Y = np.meshgrid(xs, ys)                      # (Ny, Nx)，与 FEM 节点布置一致
    uxF = res_f["modes_ux"][cfg.mode_index]
    uyF = res_f["modes_uy"][cfg.mode_index]
    viz.plot_fields_pair(X, Y, uxF, uyF,
                         ["FEM mode u_x", "FEM mode u_y"], cfg.block,
                         os.path.join(cfg.outdir, "fem_mode.png"))

    # 评估网格上的材料场（常数系数，预计算）
    XYg = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1))
    Sg, rg = core.material_fields(sdf, XYg, cfg)
    simp_g = core.simp_factor(Sg, cfg)
    MG = (Sg ** cfg.mask_pow_loss).squeeze(1).numpy().reshape(X.shape)   # 残差型损失的掩码
    MR = (Sg ** cfg.mask_pow_ray).squeeze(1).numpy().reshape(X.shape)    # Rayleigh 掩码
    rhog = rg.squeeze(1).numpy().reshape(X.shape)

    # ---------- 3) PINN：网络、点池、优化器 ----------
    nets = core.build_nets(cfg)
    params = [p for n in nets.values() for p in n.parameters()]
    if cfg.omega2_trainable:
        log_om2 = torch.nn.Parameter(torch.tensor(float(np.log(lam_tgt))))
        params = params + [log_om2]

    def om2_now():
        return torch.exp(log_om2) if cfg.omega2_trainable else torch.tensor(lam_tgt)

    gen = torch.Generator().manual_seed(cfg.seed + 7)
    pool = core.build_pool(sdf, cfg, cfg.n_pool_int, cfg.n_pool_blk,
                           cfg.n_pool_bnd, seed=cfg.seed)

    weights = dict(pde=cfg.w_pde, cons=cfg.w_cons, dir=cfg.w_dir,
                   neu=cfg.w_neu, norm=cfg.w_norm, blk=cfg.w_blk)
    hist = {k: [] for k in ("iter", "pde", "cons", "pde_blk", "cons_blk",
                            "dir", "neu", "norm", "ray", "R", "mass")}

    frame_state = {"id": 0, "prev": None}

    def snapshot(stage, it):
        UX, UY = core.eval_uv_grid(nets, XYg)
        UX = UX.numpy().reshape(X.shape)
        UY = UY.numpy().reshape(X.shape)
        UXa, UYa, frame_state["prev"] = viz.align_sign(UX, UY, frame_state["prev"])
        viz.save_snapshot(frame_state["id"], stage, it, X, Y, UXa, UYa,
                          cfg.block, frames_dir)
        frame_state["id"] += 1

    def log_point(it, terms, aux):
        hist["iter"].append(it)
        for k in ("pde", "cons", "pde_blk", "cons_blk", "dir", "neu", "norm", "ray"):
            hist[k].append(float(terms[k]))
        hist["R"].append(float(aux["R"]))
        hist["mass"].append(float(aux["mass"]))

    snapshot("init", 0)

    # ---------- 阶段 A：Adam（每步从池中重采样） ----------
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.adam_iters, eta_min=cfg.lr_min)
    tA = time.time()
    diverged = False
    for it in range(1, cfg.adam_iters + 1):
        b = core.minibatch(pool, cfg, gen)
        terms, aux = core.compute_losses(nets, b, cfg, om2_now(), lam_tgt)
        w_ray_eff = cfg.w_ray if it >= cfg.rayleigh_warmup else 0.0
        if cfg.adaptive_weights and it % cfg.adapt_every == 0:
            core.update_adaptive_weights(terms, weights, params, cfg)
        loss = core.total_loss(terms, weights, w_ray_eff)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if it == 1 or it % cfg.log_every == 0:
            log_point(it, terms, aux)
            print(f"[Adam {it:5d}] total={float(loss):.3e}"
                  f" pde={float(terms['pde']):.2e} cons={float(terms['cons']):.2e}"
                  f" dir={float(terms['dir']):.2e} neu={float(terms['neu']):.2e}"
                  f" norm={float(terms['norm']):.2e} ray={float(terms['ray']):.2e}"
                  f" R={float(aux['R']):.5f} mass={float(aux['mass']):.3f}"
                  f" lr={sched.get_last_lr()[0]:.1e}")
        if it % cfg.snapshot_every == 0:
            snapshot("Adam", it)
        if not torch.isfinite(loss):
            print("[警告] Adam 阶段损失非有限，提前终止。建议：增大 rayleigh_warmup 或减小 lr。")
            diverged = True
            break
    print(f"[Adam] 完成 ({time.time()-tA:.1f}s)")
    if cfg.save_ckpt:
        core.save_checkpoint(os.path.join(cfg.outdir, "model_adam.pt"), cfg, sdf, nets,
                             extra=dict(stage="adam", lam_tgt=lam_tgt))

    # ---------- 阶段 B：L-BFGS（固定点集、全批量、分块保存快照） ----------
    lb = {"blk": [], "train": [], "val": []}
    if not diverged:
        fb = core.build_pool(sdf, cfg, cfg.lbfgs_int, cfg.lbfgs_blk,
                             cfg.lbfgs_bnd, seed=cfg.seed + 1)
        # 独立验证点集：完全不参与训练，用来看"损失还在降"是不是只是在拟合那批固定配点
        vb = core.build_pool(sdf, cfg, cfg.lbfgs_val_int, cfg.lbfgs_blk,
                             cfg.lbfgs_bnd, seed=cfg.seed + 4021)
        opt2 = torch.optim.LBFGS(params, max_iter=cfg.lbfgs_max_iter_per,
                                 history_size=50, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-12, tolerance_change=1e-14)
        it_now = cfg.adam_iters

        def closure():
            opt2.zero_grad()
            terms_c, aux_c = core.compute_losses(nets, fb, cfg, om2_now(), lam_tgt)
            loss_c = core.total_loss(terms_c, weights, cfg.w_ray)
            loss_c.backward()
            closure.last = (terms_c, aux_c, loss_c)
            return loss_c

        best = dict(val=float("inf"), blk=-1, state=None)
        prev, stall = None, 0
        tB = time.time()
        for k in range(cfg.lbfgs_outer):
            opt2.step(closure)
            terms, aux, loss = closure.last
            it_now += cfg.lbfgs_max_iter_per
            log_point(it_now, terms, aux)
            cur = float(loss)

            vstat, vtot = core.eval_losses(nets, vb, cfg, om2_now(), lam_tgt,
                                           weights, cfg.w_ray)
            lb["blk"].append(k + 1)
            lb["train"].append(cur)
            lb["val"].append(vtot)
            gap = vtot / (cur + 1e-300)
            print(f"[LBFGS 块 {k+1}/{cfg.lbfgs_outer}] train={cur:.3e} val={vtot:.3e}"
                  f" (val/train={gap:.2f}) R={float(aux['R']):.6f}"
                  f" mass={float(aux['mass']):.4f}")
            snapshot("LBFGS", it_now)

            if vtot < best["val"]:
                best = dict(val=vtot, blk=k + 1,
                            state={n: copy.deepcopy(m.state_dict()) for n, m in nets.items()})
            if not torch.isfinite(loss):
                print("[警告] L-BFGS 出现非有限损失，停止精调（保留当前参数）。")
                break
            rel = 1.0 if prev is None else (prev - cur) / max(abs(prev), 1e-300)
            stall = stall + 1 if rel < cfg.lbfgs_stop_rtol else 0
            prev = cur
            if stall >= cfg.lbfgs_patience:
                print(f"[LBFGS] 连续 {stall} 块相对改善 < {cfg.lbfgs_stop_rtol:g}，"
                      f"判定收敛，提前结束（共 {k+1}/{cfg.lbfgs_outer} 块）。")
                break
        print(f"[LBFGS] 完成 ({time.time()-tB:.1f}s)")
        if cfg.lbfgs_restore_best and best["state"] is not None:
            if best["blk"] != lb["blk"][-1]:
                print(f"[LBFGS] 验证损失最优在第 {best['blk']} 块（val={best['val']:.3e}），"
                      f"回滚到该块的参数。若想保留最后一块，把 lbfgs_restore_best 设为 False。")
                for n, m in nets.items():
                    m.load_state_dict(best["state"][n])
            else:
                print(f"[LBFGS] 最后一块即验证最优（block {best['blk']}），"
                      f"说明还没到过拟合配点的阶段，可以继续加大 lbfgs_outer。")
        viz.plot_lbfgs_curve(lb["blk"], np.array(lb["train"]), np.array(lb["val"]),
                             os.path.join(cfg.outdir, "lbfgs_curve.png"))

    # ---------- 4) 最终评估与出图 ----------
    rep = core.eval_report(nets, XYg, cfg, simp=simp_g)

    def G(k):
        return rep[k].squeeze(1).numpy().reshape(X.shape)

    UX, UY, EN = G("ux"), G("uy"), G("en")
    D1s, D2s = G("div1_s"), G("div2_s")          # sigma 网络一阶导
    D1u, D2u = G("div1_u"), G("div2_u")          # u 网络二阶导
    SXXs, SYYs, SXYs = G("sxx_s"), G("syy_s"), G("sxy_s")
    SXXu, SYYu, SXYu = G("sxx_u"), G("syy_u"), G("sxy_u")

    # 节点网格上的求积权重：梯形法（边界节点 1/2、角点 1/4），再归一化成"加权平均"。
    # 注意不能直接用 np.mean：它把左右夹持边算满权重，而那里恰好是应变能最大、动能为零，
    # 会让 Rayleigh 商系统性偏高几个百分点（本例约 +4%），看起来像是 PINN 没训好。
    wx = np.ones(X.shape[1]); wx[0] = wx[-1] = 0.5
    wy = np.ones(X.shape[0]); wy[0] = wy[-1] = 0.5
    Wq = np.outer(wy, wx)
    Wq /= Wq.sum()

    def avg(F):
        return float(np.sum(Wq * F))

    # 网格求积的 Rayleigh 商（最终汇报的 PINN 特征值，不直接引用 FEM 值）
    den_g = avg(MR * rhog * (UX ** 2 + UY ** 2))
    num_g = avg(MR * EN)
    R_final = num_g / (den_g + 1e-300)
    mass_g = cfg.area * den_g

    # FEM 模态用同一加权范数归一化，供符号对齐与对拍
    wgt = MR * rhog
    massF = cfg.area * avg(wgt * (uxF ** 2 + uyF ** 2))
    cF = 1.0 / np.sqrt(massF + 1e-300)
    uxFn, uyFn = cF * uxF, cF * uyF

    # 统一质量归一化 + 符号对齐：线性量乘 a=c*sign，二次量乘 c^2（与符号无关）
    c = 1.0 / np.sqrt(mass_g + 1e-300)
    ip0 = avg(wgt * (c * UX * uxFn + c * UY * uyFn))
    a = c if ip0 >= 0 else -c
    UXn, UYn = a * UX, a * UY
    ENn = c * c * EN
    D1sn, D2sn = a * D1s, a * D2s
    D1un, D2un = a * D1u, a * D2u
    SIG_s = (a * SXXs, a * SYYs, a * SXYs)
    SIG_u = (a * SXXu, a * SYYu, a * SXYu)

    # 残差 r = div sigma + omega^2 rho u，两条路线各算一次
    R1s = D1sn + lam_tgt * rhog * UXn
    R2s = D2sn + lam_tgt * rhog * UYn
    R1u = D1un + lam_tgt * rhog * UXn
    R2u = D2un + lam_tgt * rhog * UYn

    # q 的两种写法：能量式（u 一阶导）与强形式（u 二阶导），积分意义下相等、逐点不等
    Q = ENn - R_final * rhog * (UXn ** 2 + UYn ** 2)
    Q_strong = -(D1un * UXn + D2un * UYn) - R_final * rhog * (UXn ** 2 + UYn ** 2)

    # 对拍指标
    ip = avg(wgt * (UXn * uxFn + UYn * uyFn))
    num_mac = ip ** 2
    den_mac = (avg(wgt * (UXn ** 2 + UYn ** 2))
               * avg(wgt * (uxFn ** 2 + uyFn ** 2)) + 1e-300)
    mac = num_mac / den_mac
    rel_l2 = np.sqrt(avg(wgt * ((UXn - uxFn) ** 2 + (UYn - uyFn) ** 2))
                     / (avg(wgt * (uxFn ** 2 + uyFn ** 2)) + 1e-300))
    rel_lam = abs(R_final - lam_tgt) / lam_tgt
    rel_lam_rich = abs(R_final - lam_rich) / lam_rich

    # 积分校核：int S^2 Ae(u):e(u) = R，int S^2 rho|u|^2 = 1，int S^2 q = 0
    I_en = cfg.area * avg(MR * ENn)
    I_kin = cfg.area * avg(MR * rhog * (UXn ** 2 + UYn ** 2))
    I_q = cfg.area * avg(MR * Q)
    I_qs = cfg.area * avg(MR * Q_strong)

    # 两条导数路线的自洽性
    rms_du = np.sqrt(avg(MG * (D1un ** 2 + D2un ** 2)))
    rms_dd = np.sqrt(avg(MG * ((D1sn - D1un) ** 2 + (D2sn - D2un) ** 2)))
    div_rel = rms_dd / (rms_du + 1e-300)
    rms_rs = np.sqrt(avg(MG * (R1s ** 2 + R2s ** 2)))
    rms_ru = np.sqrt(avg(MG * (R1u ** 2 + R2u ** 2)))

    # 残差下限：omega^2 被钉在一个有离散误差的锚点上，即使 u 完全正确，
    # r = (lam_target - lam_exact) * rho * u 也降不下去。以此判断"还该不该再练"。
    rms_rho_u = np.sqrt(avg(MG * rhog ** 2 * (UXn ** 2 + UYn ** 2)))
    lam_err = (0.25 * abs(lam_f - lam_rich) if cfg.use_richardson_target
               else abs(lam_tgt - lam_rich))
    res_floor = lam_err * rms_rho_u
    if rms_rs < 3.0 * res_floor:
        hint = ("已接近锚点离散误差造成的下限：再加 L-BFGS 轮数收益有限，"
                "想继续降请开 use_richardson_target=True 或加密 FEM 网格")
    else:
        hint = ("离下限还有距离：加大 lbfgs_outer（并相应加大 lbfgs_int）通常还能继续降")

    # ---- FEM 版本的 q：把 FEM 模态代回单元，用与 K、M 同一套 Q4 形函数算导数 ----
    # 积分用 2x2 Gauss，故 I_en = phi^T K phi、I_kin = phi^T M phi，I_q 必为机器零，
    # 这是 FEM 侧自带的校核；PINN 的 q 则在同一批单元中心上重新评估一次，二者可逐点相减。
    # 用原始 M-归一化模态（phi^T M phi = 1）算积分，I_en 才严格等于 lam_FEM；
    # q 是二次量，画图时再统一乘 cF^2 换成与 PINN 相同的质量归一化即可。
    qf = fem.mode_q_fields(cfg.nx_fem, cfg.ny_fem, cfg.Lx, cfg.Ly, cfg.E, cfg.nu,
                           uxF, uyF, dens_fn, stiff_fn, lam_f)
    Xe, Ye = qf["xc"], qf["yc"]
    Q_fem = cF ** 2 * qf["q"]
    XYe = torch.tensor(np.stack([Xe.ravel(), Ye.ravel()], axis=1))
    Se_t, rho_e_t = core.material_fields(sdf, XYe, cfg)
    rho_e = rho_e_t.squeeze(1).numpy().reshape(Xe.shape)
    rep_e = core.eval_report(nets, XYe, cfg, simp=core.simp_factor(Se_t, cfg))

    def Ge(k):
        return rep_e[k].squeeze(1).numpy().reshape(Xe.shape)

    # 用与节点网格同一组缩放/符号因子，两套场才可比
    UXe, UYe = a * Ge("ux"), a * Ge("uy")
    Q_pinn = c * c * Ge("en") - R_final * rho_e * (UXe ** 2 + UYe ** 2)
    dq = Q_pinn - Q_fem
    q_rel = float(np.sqrt(np.mean(dq ** 2)) / (np.sqrt(np.mean(Q_fem ** 2)) + 1e-300))
    je_c = int(np.argmin(np.abs(Ye[:, 0] - 0.5 * (cfg.block[2] + cfg.block[3]))))

    viz.plot_fields_pair(X, Y, UXn, UYn,
                         ["PINN final u_x (mass-normalized)",
                          "PINN final u_y (mass-normalized)"],
                         cfg.block, os.path.join(cfg.outdir, "final_fields.png"))
    viz.plot_fields_pair(X, Y, np.abs(R1s), np.abs(R2s),
                         ["|PDE residual r_x|", "|PDE residual r_y|"],
                         cfg.block, os.path.join(cfg.outdir, "residual_maps.png"),
                         sym=False)
    viz.plot_compare_fem(X, Y, UXn, UYn, uxFn, uyFn, cfg.block,
                         os.path.join(cfg.outdir, "compare_fem.png"))
    viz.plot_stress_compare(X, Y, SIG_s, SIG_u, cfg.block,
                            os.path.join(cfg.outdir, "stress_compare.png"))
    viz.plot_div_compare(X, Y, (D1sn, D2sn), (D1un, D2un), cfg.block,
                         os.path.join(cfg.outdir, "div_compare.png"))
    viz.plot_residual_compare(X, Y, (R1s, R2s), (R1u, R2u), cfg.block,
                              os.path.join(cfg.outdir, "residual_compare.png"))
    viz.plot_q_surface(X, Y, Q, cfg.block, R_final,
                       os.path.join(cfg.outdir, "q_surface_3d.png"))
    viz.plot_q_surface(X, Y, Q_strong, cfg.block, R_final,
                       os.path.join(cfg.outdir, "q_surface_strong_3d.png"),
                       label=r"$\tilde q=-(\nabla\!\cdot\!Ae(u))\cdot u-\omega^2\rho|u|^2$")
    viz.plot_q_surface(Xe, Ye, Q_fem, cfg.block, lam_f,
                       os.path.join(cfg.outdir, "q_surface_fem_3d.png"),
                       label=r"FEM: $q=Ae(u_h){:}e(u_h)-\omega^2\rho|u_h|^2$")
    viz.plot_q_compare(Xe, Ye, Q_pinn, Q_fem, cfg.block, R_final, lam_f,
                       os.path.join(cfg.outdir, "q_compare.png"), jc=je_c)
    viz.plot_loss_curves(hist, lam_tgt, os.path.join(cfg.outdir, "loss_curves.png"))
    viz.make_gif(frames_dir, os.path.join(cfg.outdir, "evolution.gif"))
    np.savez(os.path.join(cfg.outdir, "history.npz"),
             **{k: np.array(v) for k, v in hist.items()},
             lbfgs_blk=np.array(lb["blk"]), lbfgs_train=np.array(lb["train"]),
             lbfgs_val=np.array(lb["val"]))
    np.savez(os.path.join(cfg.outdir, "final_fields.npz"),
             X=X, Y=Y, ux=UXn, uy=UYn, ux_fem=uxFn, uy_fem=uyFn,
             q=Q, q_strong=Q_strong, en=ENn,
             xe=Xe, ye=Ye, q_pinn_elem=Q_pinn, q_fem_elem=Q_fem,
             en_fem_elem=cF ** 2 * qf["en"], kin_fem_elem=cF ** 2 * qf["kin"],
             r1=R1s, r2=R2s, r1_u=R1u, r2_u=R2u,
             div1_s=D1sn, div2_s=D2sn, div1_u=D1un, div2_u=D2un,
             sxx_s=SIG_s[0], syy_s=SIG_s[1], sxy_s=SIG_s[2],
             sxx_u=SIG_u[0], syy_u=SIG_u[1], sxy_u=SIG_u[2],
             mask=MG, mask_ray=MR, rho=rhog, lam_fem=lam_f, lam_fem_coarse=lam_c,
             lam_richardson=lam_rich, lam_target=lam_tgt, lam_pinn=R_final)

    # ---------- 5) 保存网络权重 ----------
    if cfg.save_ckpt:
        p = core.save_checkpoint(
            os.path.join(cfg.outdir, "model_final.pt"), cfg, sdf, nets,
            extra=dict(stage="final", lam_fem=lam_f, lam_fem_coarse=lam_c,
                       lam_richardson=lam_rich, lam_target=lam_tgt,
                       R_final=R_final, mac=mac, rel_l2=rel_l2, mass=mass_g))
        print(f"[保存] 网络权重 -> {p}"
              f"（载入：pinn_core.load_checkpoint(路径)，注意先 set_default_dtype(float64)）")

    dt = time.time() - t_start
    lines = [
        "==================== 结果摘要 ====================",
        f"omega^2_FEM  粗网格({cfg.nx_fem_coarse}x{cfg.ny_fem_coarse}) = {lam_c:.6f}",
        f"omega^2_FEM  细网格({cfg.nx_fem}x{cfg.ny_fem}) = {lam_f:.6f}"
        f"   (粗细相对差 {mesh_rel:.2e})",
        f"omega^2_Richardson 外推 = {lam_rich:.6f}   (更接近连续解)",
        f"训练锚点 omega^2_target = {lam_tgt:.6f}",
        f"omega^2_PINN = R(u_final) = {R_final:.6f}",
        f"特征值相对误差 |R-lam_target|/lam_target = {rel_lam:.3e}"
        f"   (对 Richardson: {rel_lam_rich:.3e})",
        f"模态场相对 L2 误差（S^2*rho 加权，符号对齐后） = {rel_l2:.3e}",
        f"MAC = {mac:.6f}",
        f"PDE 残差 RMS（S^2 加权）：sigma 网络路线 = {rms_rs:.3e}，"
        f"u 二阶导路线 = {rms_ru:.3e}",
        f"锚点造成的残差下限估计 ≈ {res_floor:.2e}"
        f"  ( |lam_target-lam_exact|≈{lam_err:.2e} 乘 RMS(rho|u|)={rms_rho_u:.2f} )",
        f"  -> {hint}",
        f"两条路线的散度自洽性 ||div sigma^NN - div Ae(u)|| / ||div Ae(u)|| = {div_rel:.3e}",
        f"掩码口径：损失用 S^{cfg.mask_pow_loss:g}，Rayleigh 用 S^{cfg.mask_pow_ray:g}；"
        f"刚度缩放 use_simp={cfg.use_simp}"
        + (f"(S^{cfg.simp_p})" if cfg.use_simp else "(FEM 与 PINN 均不缩放)"),
        f"积分校核：int S^2 Ae(u):e(u) = {I_en:.6f}（应=R）,"
        f" int S^2 rho|u|^2 = {I_kin:.6f}（应=1）",
        f"          int S^2 q = {I_q:.3e}（应≈0）,"
        f" int S^2 q_strong = {I_qs:.3e}（应≈0）",
        f"FEM 侧 q（单元级 Gauss 求积）：int Ae:e = phi^T K phi = {qf['I_en']:.6f}"
        f"（应=lam_FEM={lam_f:.6f}）, int rho|u|^2 = {qf['I_kin']:.6f}",
        f"          int q_FEM = {qf['I_q']:.3e}（本征方程本身，应为机器零）",
        f"q 场对照（同一批单元中心）：||q_PINN - q_FEM|| / ||q_FEM|| = {q_rel:.3e}",
        f"质量积分 mass = {mass_g:.6f}（目标 1）",
        f"总耗时 {dt/60.0:.1f} min",
        "输出目录: " + os.path.abspath(cfg.outdir),
        "=================================================",
    ]
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(cfg.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    return dict(lam_pinn=R_final, lam_target=lam_tgt, mac=mac, rel_l2=rel_l2)


if __name__ == "__main__":
    main()
