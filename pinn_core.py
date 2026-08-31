"""PINN 核心：网络、材料场（SDF + tanh-Heaviside + 重块）、采样、损失。依赖 PyTorch。

设计要点：
- 混合格式：5 个独立 MLP 分别输出 ux, uy, sxx, sxy, syy，动量方程只需 sigma 的一阶导，
  本构残差把 sigma 网络与 u 网络的应变绑定，全程只用一阶自动微分；
- 坐标归一化写在网络第一层（固定仿射），外部自动微分始终针对物理坐标，损失无需改写；
- SDF 预训练后冻结，所有点池上的 S、rho 预先在 no_grad 下算好，训练时当常数用；
- 材料掩码：残差型损失（pde/cons/dir/neu）乘 S^mask_pow_loss（默认 S 的一次方），
  Rayleigh 商的分子分母与质量归一化乘 S^mask_pow_ray（默认 S^2）；
  点池里把这两种权重都预先算好，训练时当常数用。

评估阶段（eval_report）额外提供"两条路线"的对比量，供事后诊断：
  路线 A（sigma 网络，一阶导）: div sigma^NN = (d_x sxx + d_y sxy, d_x sxy + d_y syy)
  路线 B（u   网络，二阶导）: div A e(u)   = Navier 算子作用在 u 上
两者之差就是混合格式的自洽性误差（本构残差 L_cons 的"导数版"）。
"""
import copy
import math
import torch
import torch.nn as nn


# ============================== 网络 ==============================

class FeatureMap(nn.Module):
    """物理坐标 (x, y) -> 归一化坐标 [-1,1]^2 的固定仿射（网络内部完成），可选 Fourier 特征。

    对外接口始终吃物理坐标，因此对输入求导得到的就是对物理坐标的导数，损失公式不用改。
    """

    def __init__(self, Lx, Ly, fourier=False, n_ff=32, sigma_ff=2.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0 / Lx, 2.0 / Ly]))
        self.register_buffer("shift", torch.tensor([1.0, 1.0]))
        self.fourier = fourier
        if fourier:
            self.register_buffer("B", torch.randn(2, n_ff) * sigma_ff)
            self.out_dim = 2 + 2 * n_ff
        else:
            self.out_dim = 2

    def forward(self, xy):
        z = xy * self.scale - self.shift
        if self.fourier:
            zb = 2.0 * math.pi * (z @ self.B)
            z = torch.cat([z, torch.sin(zb), torch.cos(zb)], dim=1)
        return z


class MLP(nn.Module):
    def __init__(self, fmap, hidden=64, layers=3):
        super().__init__()
        self.fmap = fmap
        dims = [fmap.out_dim] + [hidden] * layers + [1]
        self.lin = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, xy):
        z = self.fmap(xy)
        for L in self.lin[:-1]:
            z = torch.tanh(L(z))
        return self.lin[-1](z)


def build_nets(cfg):
    """5 个独立网络（共享同一个特征映射实例，特征映射本身无可训练参数）。"""
    fmap = FeatureMap(cfg.Lx, cfg.Ly, cfg.fourier, cfg.n_ff, cfg.sigma_ff)
    return {name: MLP(fmap, cfg.hidden, cfg.layers)
            for name in ("ux", "uy", "sxx", "sxy", "syy")}


# ============================== SDF 与材料场 ==============================

def build_sdf(cfg):
    return MLP(FeatureMap(cfg.Lx, cfg.Ly, fourier=False), cfg.hidden, cfg.layers)


def analytic_sdf(xy, cfg):
    """预训练的目标场：按 cfg.sdf_init 给出解析的（符号）距离函数。

    约定 phi > 0 有材料、phi < 0 无材料，phi = 0 是材料/空洞界面。

    - "const"     : phi ≡ sdf_target。全域满材料，但 |grad phi| = 0，
                    严格说不是距离函数；本阶段与 FEM 对拍满材料板时最干净。
    - "edge_dist" : phi = min(x, Lx-x, y, Ly-y) + offset。
                    对矩形内点，四条边的垂距取最小值就是到 partial-Omega 的最短距离，
                    所以这是真正的距离函数，|grad phi| = 1（中轴线上不可导）。
                    offset = 0 时零水平集正好落在 partial-Omega 上 => 边界 S = 0.5；
                    offset >= 5*eps 时零水平集被推到域外 => 边界也是满材料，
                    这恰好就是"初始设计 = 整块实体"的几何含义。
    - "holes"     : 再与若干播种圆孔的符号距离 |x-c| - r 取小。孔内为负、孔外为正，
                    1-Lipschitz 函数取小仍是 1-Lipschitz，故整体仍满足 |grad phi| = 1。
    """
    x, y = xy[:, 0:1], xy[:, 1:2]
    if cfg.sdf_init == "const":
        return torch.full_like(x, float(cfg.sdf_target))
    d = torch.minimum(torch.minimum(x, cfg.Lx - x),
                      torch.minimum(y, cfg.Ly - y)) + cfg.sdf_offset
    if cfg.sdf_init == "edge_dist":
        return d
    if cfg.sdf_init == "holes":
        for (cx, cy, r) in cfg.sdf_holes:
            d = torch.minimum(d, torch.sqrt((x - cx) ** 2 + (y - cy) ** 2 + 1e-30) - r)
        return d
    raise ValueError(f"未知的 sdf_init: {cfg.sdf_init!r}，可选 const / edge_dist / holes")


def sdf_band_xy(n, cfg, gen):
    """零水平集邻域（"边界带"）采样：域四边的内侧窄带 + holes 初始化时各播种孔的孔缘窄带。

    S = H(phi/eps) 只对 |phi| <~ 4*eps 的窄带敏感，这条带正是拟合误差最要命的地方，
    所以预训练对它单独采样、单独加权。细节：
      - 四边按边长比例分配点数，隔一个点严格压在边上（d=0），其余在 (0, band] 内侧；
      - 额外附上 4 个角点（两条边的距离函数在角上交汇，误差常在这里冒尖）；
      - holes 模式再对每个孔沿圆周均匀取角度，径向 r + U[-band, band]，同样隔点压在圆上。
    所有点都在闭域 [0,Lx]x[0,Ly] 内（网络只会在域内被查询）。n<=0 时返回空点集。
    """
    if n <= 0:
        return torch.zeros(0, 2)
    band = float(cfg.sdf_bnd_band)
    Lx, Ly = cfg.Lx, cfg.Ly
    use_holes = (cfg.sdf_init == "holes") and len(cfg.sdf_holes) > 0
    n_edge = n - (n // 2 if use_holes else 0)
    n_hole = n - n_edge

    pts = []
    per = 2.0 * (Lx + Ly)
    counts = [int(round(n_edge * q / per)) for q in (Lx, Lx, Ly, Ly)]
    counts[0] += n_edge - sum(counts)
    for c, side in zip(counts, ("bot", "top", "left", "right")):
        if c <= 0:
            continue
        t = torch.rand(c, 1, generator=gen)
        d = torch.rand(c, 1, generator=gen) * band
        d[::2] = 0.0                                   # 隔一个点严格压在边上
        if side == "bot":
            p = torch.cat([t * Lx, d], dim=1)
        elif side == "top":
            p = torch.cat([t * Lx, Ly - d], dim=1)
        elif side == "left":
            p = torch.cat([d, t * Ly], dim=1)
        else:
            p = torch.cat([Lx - d, t * Ly], dim=1)
        pts.append(p)
    pts.append(torch.tensor([[0.0, 0.0], [Lx, 0.0], [0.0, Ly], [Lx, Ly]]))

    if use_holes and n_hole > 0:
        m0 = n_hole // len(cfg.sdf_holes)
        for i, (cx, cy, r) in enumerate(cfg.sdf_holes):
            m = m0 + (n_hole - m0 * len(cfg.sdf_holes) if i == 0 else 0)
            th = torch.rand(m, 1, generator=gen) * (2.0 * math.pi)
            dr = (2.0 * torch.rand(m, 1, generator=gen) - 1.0) * band
            dr[::2] = 0.0                              # 隔一个点严格压在孔缘上
            rr = (r + dr).clamp_min(1e-6)
            pts.append(torch.cat([cx + rr * torch.cos(th),
                                  cy + rr * torch.sin(th)], dim=1))

    p = torch.cat(pts, dim=0)
    p[:, 0].clamp_(0.0, Lx)
    p[:, 1].clamp_(0.0, Ly)
    return p


def pretrain_sdf(cfg):
    """把 SDF 网络拟合到 analytic_sdf 给出的目标场，然后冻结。两阶段：

    A) Adam：每步 2048 个域内均匀点 + cfg.sdf_batch_bnd 个边界带点（sdf_band_xy），
       损失 = mse_int + sdf_w_bnd * mse_bnd。均匀采样下边界带的点数占比只有 ~band/面积，
       而 S 恰恰只在那条带里对误差敏感，所以必须显式加采样、加权。
    B) L-BFGS 微调：固定点集（Sobol 内部 + 边界带）上全批量强 Wolfe 线搜索，
       分块迭代（sdf_lbfgs_outer * sdf_lbfgs_max_iter_per），独立验证集只评估不训练，
       连续 sdf_lbfgs_patience 块相对改善 < sdf_lbfgs_stop_rtol 判停滞提前结束，
       结束时回滚到验证损失最优的权重（防过拟合固定配点）。sdf_lbfgs_outer=0 退回纯 Adam。

    返回 (sdf, 末次组合损失, 逐次组合损失列表)；列表先记 Adam 每步、再记 L-BFGS 每次
    闭包评估（含线搜索），画出来就是一条完整收敛曲线。若跑了 L-BFGS，"末次组合损失"
    取验证集上的最优值，比训练集数字更诚实。
    采样使用独立 torch.Generator(seed=cfg.seed+31)，不消耗全局 RNG，结果与调用顺序无关。
    非常数目标（edge_dist / holes）在中轴线、孔边有折线/曲率，tanh 网络会抹平一点，
    但那些位置 |phi| >> eps，S 已经饱和，抹平无害；关键的带内误差在诊断里会打印。
    """
    sdf = build_sdf(cfg)
    gen = torch.Generator().manual_seed(cfg.seed + 31)
    LxLy = torch.tensor([cfg.Lx, cfg.Ly])
    hist = []

    def combined(xi, ti, xb, tb):
        """组合损失及其分量：(总, 内部 mse, 边界带 mse)。"""
        li = ((sdf(xi) - ti) ** 2).mean()
        if xb.shape[0] == 0:
            return li, li, li * 0.0
        lb = ((sdf(xb) - tb) ** 2).mean()
        return li + cfg.sdf_w_bnd * lb, li, lb

    # ---------- 阶段 A：Adam（每步重采样） ----------
    opt = torch.optim.Adam(sdf.parameters(), lr=cfg.sdf_pretrain_lr)
    loss = torch.tensor(0.0)
    for _ in range(cfg.sdf_pretrain_iters):
        xi = torch.rand(2048, 2, generator=gen) * LxLy
        xb = sdf_band_xy(cfg.sdf_batch_bnd, cfg, gen)
        with torch.no_grad():
            ti = analytic_sdf(xi, cfg)
            tb = analytic_sdf(xb, cfg)
        loss, _, _ = combined(xi, ti, xb, tb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach()))
    final = float(loss.detach())

    # ---------- 阶段 B：L-BFGS 微调（固定点集 + 独立验证集） ----------
    if cfg.sdf_lbfgs_outer > 0 and cfg.sdf_lbfgs_max_iter_per > 0:
        xi = sobol_xy(cfg.sdf_lbfgs_int, cfg, seed=cfg.seed + 101)
        xb = sdf_band_xy(cfg.sdf_lbfgs_bnd, cfg, gen)
        xiv = sobol_xy(max(cfg.sdf_lbfgs_int // 2, 1024), cfg, seed=cfg.seed + 4021)
        xbv = sdf_band_xy(max(cfg.sdf_lbfgs_bnd // 2, 256), cfg, gen)
        with torch.no_grad():
            ti, tb = analytic_sdf(xi, cfg), analytic_sdf(xb, cfg)
            tiv, tbv = analytic_sdf(xiv, cfg), analytic_sdf(xbv, cfg)

        opt2 = torch.optim.LBFGS(sdf.parameters(), max_iter=cfg.sdf_lbfgs_max_iter_per,
                                 history_size=50, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-14, tolerance_change=1e-16)

        def closure():
            opt2.zero_grad()
            l, li, lb = combined(xi, ti, xb, tb)
            l.backward()
            hist.append(float(l.detach()))
            closure.last = (float(l.detach()), float(li.detach()), float(lb.detach()))
            return l

        best = dict(val=float("inf"), state=None, blk=-1)
        prev, stall = None, 0
        for k in range(cfg.sdf_lbfgs_outer):
            opt2.step(closure)
            cur, cur_i, cur_b = closure.last
            with torch.no_grad():
                v, _, _ = combined(xiv, tiv, xbv, tbv)
            v = float(v)
            print(f"[SDF-LBFGS 块 {k + 1}/{cfg.sdf_lbfgs_outer}] train={cur:.3e}"
                  f" (int={cur_i:.2e} bnd={cur_b:.2e}) val={v:.3e}")
            if math.isfinite(v) and v < best["val"]:
                best = dict(val=v, state=copy.deepcopy(sdf.state_dict()), blk=k + 1)
            if not math.isfinite(cur):
                print("[SDF-LBFGS] 出现非有限损失，停止微调（回滚到验证最优权重）。")
                break
            rel = 1.0 if prev is None else (prev - cur) / max(abs(prev), 1e-300)
            stall = stall + 1 if rel < cfg.sdf_lbfgs_stop_rtol else 0
            prev = cur
            if stall >= cfg.sdf_lbfgs_patience:
                print(f"[SDF-LBFGS] 连续 {stall} 块相对改善 < {cfg.sdf_lbfgs_stop_rtol:g}，"
                      f"判定收敛，提前结束（共 {k + 1}/{cfg.sdf_lbfgs_outer} 块）。")
                break
        if best["state"] is not None:
            sdf.load_state_dict(best["state"])
            final = best["val"]

    for p in sdf.parameters():
        p.requires_grad_(False)
    sdf.eval()
    return sdf, final, hist


def sdf_grad_norm(sdf, xy):
    """|grad phi|，形状 (N,1)。真正的距离函数应处处 ≈ 1（中轴线附近会被网络抹平而偏小），
    常数初始化则 ≈ 0 —— 这是判断"到底是不是距离函数"最直接的一张图。"""
    p = xy.clone().requires_grad_(True)
    phi = sdf(p)
    g = torch.autograd.grad(phi, p, grad_outputs=torch.ones_like(phi))[0]
    return torch.sqrt((g ** 2).sum(dim=1, keepdim=True) + 1e-30).detach()


def heaviside_tanh(phi, eps):
    return 0.5 * (1.0 + torch.tanh(phi / eps))


def chi_block(xy, block, w):
    """重块的光滑指示函数（与 fem_baseline.analytic_chi_block 完全一致的 torch 版）。"""
    x, y = xy[:, 0:1], xy[:, 1:2]
    bx = 0.5 * (torch.tanh((x - block[0]) / w) + torch.tanh((block[1] - x) / w))
    by = 0.5 * (torch.tanh((y - block[2]) / w) + torch.tanh((block[3] - y) / w))
    return bx * by


@torch.no_grad()
def material_fields_full(sdf, xy, cfg):
    """诊断用：一次返回 (phi, S, chi, S_eff, rho)，形状均为 (N,1)。

    phi   : SDF 网络原始输出
    S     : Heaviside(phi) in (0,1)
    chi   : 重块光滑指示函数
    S_eff : S + (1-S)*chi     重块区强制为 1（不可优化）
    rho   : S_eff * (1 + (gamma-1)*chi)
    """
    phi = sdf(xy)
    S = heaviside_tanh(phi, cfg.eps_heaviside)
    chi = chi_block(xy, cfg.block, cfg.block_smooth_w)
    S_eff = S + (1.0 - S) * chi
    rho = S_eff * (1.0 + (cfg.gamma_block - 1.0) * chi)
    return phi, S, chi, S_eff, rho


def simp_factor(S_eff, cfg):
    """刚度缩放因子：use_simp=True 时为 S_eff^simp_p，否则恒为 1。

    FEM 侧的 stiff_fn 用同一个式子，两边必须一致，否则解的是两个不同的方程。
    """
    if not cfg.use_simp:
        return torch.ones_like(S_eff)
    return S_eff ** cfg.simp_p


@torch.no_grad()
def material_fields(sdf, xy, cfg):
    """返回 (S_eff, rho)，形状 (N,1)。

    重块区经光滑指示函数强制 S=1、rho=gamma（不可优化）；其余处 S 来自 SDF+Heaviside。
    SDF 已冻结，材料场是常数系数场，no_grad 预计算即可。
    """
    _, _, _, S_eff, rho = material_fields_full(sdf, xy, cfg)
    return S_eff, rho


# ============================== 采样 ==============================

def sobol_xy(n, cfg, seed):
    eng = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    p = eng.draw(n).to(torch.get_default_dtype())
    return p * torch.tensor([cfg.Lx, cfg.Ly])


def block_xy(n, cfg):
    """重块及其外扩 margin 邻域内的均匀随机点（用于局部加密强调）。"""
    m = cfg.block_sample_margin
    lo = torch.tensor([cfg.block[0] - m, cfg.block[2] - m])
    hi = torch.tensor([cfg.block[1] + m, cfg.block[3] + m])
    return lo + torch.rand(n, 2) * (hi - lo)


def boundary_xy(n_per_side, cfg):
    """返回 (dirichlet 点集[左+右], neumann 点集[下+上])。"""
    t = torch.rand(n_per_side, 1) * cfg.Ly
    left = torch.cat([torch.zeros_like(t), t], dim=1)
    t = torch.rand(n_per_side, 1) * cfg.Ly
    right = torch.cat([torch.full_like(t, cfg.Lx), t], dim=1)
    t = torch.rand(n_per_side, 1) * cfg.Lx
    bot = torch.cat([t, torch.zeros_like(t)], dim=1)
    t = torch.rand(n_per_side, 1) * cfg.Lx
    top = torch.cat([t, torch.full_like(t, cfg.Ly)], dim=1)
    return torch.cat([left, right], dim=0), torch.cat([bot, top], dim=0)


def build_pool(sdf, cfg, n_int, n_blk, n_bnd_side, seed):
    """构建点池并预计算材料场（两种掩码与 rho 都当常数）。

    mask     = S^mask_pow_loss  -> pde / cons / dir / neu 等残差型损失
    mask_ray = S^mask_pow_ray   -> Rayleigh 商的分子分母、质量归一化
    """
    pl, pr = cfg.mask_pow_loss, cfg.mask_pow_ray
    xi = sobol_xy(n_int, cfg, seed)
    xb = block_xy(n_blk, cfg)
    xd, xn = boundary_xy(n_bnd_side, cfg)
    Si, ri = material_fields(sdf, xi, cfg)
    Sb, rb = material_fields(sdf, xb, cfg)
    Sd, _ = material_fields(sdf, xd, cfg)
    Sn, _ = material_fields(sdf, xn, cfg)
    return dict(int_xy=xi, int_mask=Si ** pl, int_mask_ray=Si ** pr, int_rho=ri,
                int_simp=simp_factor(Si, cfg),
                blk_xy=xb, blk_mask=Sb ** pl, blk_mask_ray=Sb ** pr, blk_rho=rb,
                blk_simp=simp_factor(Sb, cfg),
                dir_xy=xd, dir_mask=Sd ** pl,
                neu_xy=xn, neu_mask=Sn ** pl)


def minibatch(pool, cfg, gen):
    """从点池随机取一个小批量（Adam 阶段每步重取）。"""
    out = {}
    i = torch.randint(0, pool["int_xy"].shape[0], (cfg.batch_int,), generator=gen)
    for k in ("int_xy", "int_mask", "int_mask_ray", "int_rho", "int_simp"):
        out[k] = pool[k][i]
    i = torch.randint(0, pool["blk_xy"].shape[0], (cfg.batch_blk,), generator=gen)
    for k in ("blk_xy", "blk_mask", "blk_mask_ray", "blk_rho", "blk_simp"):
        out[k] = pool[k][i]
    i = torch.randint(0, pool["dir_xy"].shape[0], (2 * cfg.batch_bnd,), generator=gen)
    out["dir_xy"], out["dir_mask"] = pool["dir_xy"][i], pool["dir_mask"][i]
    i = torch.randint(0, pool["neu_xy"].shape[0], (2 * cfg.batch_bnd,), generator=gen)
    out["neu_xy"], out["neu_mask"] = pool["neu_xy"][i], pool["neu_mask"][i]
    return out


# ============================== 物理与损失 ==============================

def _grad(f, xy):
    """f: (N,1)，返回 (df/dx, df/dy) 各 (N,1)。"""
    g = torch.autograd.grad(f, xy, grad_outputs=torch.ones_like(f), create_graph=True)[0]
    return g[:, 0:1], g[:, 1:2]


def stress_from_strain(exx, eyy, exy, E, nu):
    """平面应力本构：返回 (sxx, syy, sxy)。注意 sxy = E/(1+nu) * exy = 2 mu exy。"""
    c = E / (1.0 - nu ** 2)
    return c * (exx + nu * eyy), c * (eyy + nu * exx), (E / (1.0 + nu)) * exy


def interior_terms(nets, xy0, mask, mask_ray, rho, simp, cfg, om2):
    """内部点上的各项：PDE 残差、本构残差、Rayleigh 商的分子/分母（均为 batch 均值）。

    mask     : 残差型损失的材料掩码（默认 S 的一次方）
    mask_ray : Rayleigh 商用的掩码（默认 S^2）
    simp     : 刚度缩放 S^p（use_simp=False 时恒为 1），与 FEM 的 stiff_fn 同源
    """
    xy = xy0.clone().requires_grad_(True)
    ux = nets["ux"](xy)
    uy = nets["uy"](xy)
    ux_x, ux_y = _grad(ux, xy)
    uy_x, uy_y = _grad(uy, xy)
    exx, eyy, exy = ux_x, uy_y, 0.5 * (ux_y + uy_x)
    sxx_u, syy_u, sxy_u = stress_from_strain(exx, eyy, exy, cfg.E * simp, cfg.nu)

    sxx = nets["sxx"](xy)
    syy = nets["syy"](xy)
    sxy = nets["sxy"](xy)
    sxx_x, _ = _grad(sxx, xy)
    sxy_x, sxy_y = _grad(sxy, xy)
    _, syy_y = _grad(syy, xy)

    # 动量方程残差：div(sigma) + om2 * rho * u = 0（时谐，e^{i omega t} 约定）
    r1 = sxx_x + sxy_y + om2 * rho * ux
    r2 = sxy_x + syy_y + om2 * rho * uy
    L_pde = (mask * (r1 ** 2 + r2 ** 2)).mean()

    # 本构残差：把 sigma 网络绑定到 u 网络的应变上
    L_cons = (mask * ((sxx - sxx_u) ** 2 + (syy - syy_u) ** 2 + (sxy - sxy_u) ** 2)).mean()

    # Rayleigh 商用能量密度 Ae(u):e(u) = sigma(u):eps(u)（由 u 网络自洽计算），掩码用 S^2
    en = sxx_u * exx + syy_u * eyy + 2.0 * sxy_u * exy
    num = (mask_ray * en).mean()
    den = (mask_ray * rho * (ux ** 2 + uy ** 2)).mean()
    return L_pde, L_cons, num, den


def compute_losses(nets, b, cfg, om2, lam_fem):
    """返回 (terms 字典, 辅助量字典)。terms 均为标量张量。"""
    L_pde, L_cons, num, den = interior_terms(
        nets, b["int_xy"], b["int_mask"], b["int_mask_ray"], b["int_rho"],
        b["int_simp"], cfg, om2)
    L_pde_b, L_cons_b, _, _ = interior_terms(
        nets, b["blk_xy"], b["blk_mask"], b["blk_mask_ray"], b["blk_rho"],
        b["blk_simp"], cfg, om2)

    R = num / (den + 1e-30)                      # Rayleigh 商（面积因子约掉）
    mass = cfg.area * den                        # int S^2 rho |u|^2 dx 的蒙特卡洛估计（S^2 口径）
    L_ray = (R / lam_fem - 1.0) ** 2             # 相对形式，天然无量纲
    L_norm = (mass - 1.0) ** 2                   # 质量归一化软约束，排除平凡解

    uxd = nets["ux"](b["dir_xy"])
    uyd = nets["uy"](b["dir_xy"])
    L_dir = (b["dir_mask"] * (uxd ** 2 + uyd ** 2)).mean()

    sxyn = nets["sxy"](b["neu_xy"])
    syyn = nets["syy"](b["neu_xy"])
    L_neu = (b["neu_mask"] * (sxyn ** 2 + syyn ** 2)).mean()

    terms = dict(pde=L_pde, cons=L_cons, pde_blk=L_pde_b, cons_blk=L_cons_b,
                 dir=L_dir, neu=L_neu, norm=L_norm, ray=L_ray)

    if cfg.hard_normalization:
        # 投影式硬归一化：残差对 (u, sigma) 联合缩放是一次齐次的，
        # 因此把齐次损失统一乘 1/mass 与"先归一化再算损失"严格等价，且保持计算图简洁。
        h = 1.0 / (mass + 1e-30)
        for k in ("pde", "cons", "pde_blk", "cons_blk", "dir", "neu"):
            terms[k] = terms[k] * h
        terms["norm"] = terms["norm"] * 0.0

    return terms, dict(R=R, mass=mass)


def total_loss(terms, w, w_ray_eff):
    return (w["pde"] * terms["pde"] + w["cons"] * terms["cons"]
            + w["blk"] * (terms["pde_blk"] + terms["cons_blk"])
            + w["dir"] * terms["dir"] + w["neu"] * terms["neu"]
            + w["norm"] * terms["norm"] + w_ray_eff * terms["ray"])


def eval_losses(nets, batch, cfg, om2, lam_fem, weights, w_ray_eff):
    """只前向评估一组点上的损失（不反传），用于独立验证集监控。返回 (float 字典, total)。"""
    terms, aux = compute_losses(nets, batch, cfg, om2, lam_fem)
    tot = float(total_loss(terms, weights, w_ray_eff))
    out = {k: float(v) for k, v in terms.items()}
    out.update({k: float(v) for k, v in aux.items()})
    return out, tot


def update_adaptive_weights(terms, weights, params, cfg):
    """梯度范数平衡（Wang et al. 风格）：以 pde 项为参照调 cons/dir/neu/norm。
    只在 adaptive_weights=True 且到达 adapt_every 步时调用；ray/blk 权重保持固定。"""
    def gnorm(loss):
        gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        s = None
        for g in gs:
            if g is not None:
                s = (g ** 2).sum() if s is None else s + (g ** 2).sum()
        if s is None:
            return torch.tensor(1e-15)
        return torch.sqrt(s + 1e-30)

    nref = gnorm(terms["pde"])
    for k in ("cons", "dir", "neu", "norm"):
        tgt = float(nref / gnorm(terms[k]))
        new = (1.0 - cfg.adapt_alpha) * weights[k] + cfg.adapt_alpha * tgt
        weights[k] = float(min(max(new, 1e-2), 1e3))


# ============================== 评估 ==============================

@torch.no_grad()
def eval_uv_grid(nets, XY):
    """网格上评估位移（快照用），返回一维张量 (N,)。"""
    return nets["ux"](XY).squeeze(-1), nets["uy"](XY).squeeze(-1)


def eval_report(nets, XY0, cfg, simp=None):
    """最终评估：一次算齐所有诊断量，返回 detach 后的张量字典（形状均为 (N,1)）。

    含义（下标 _u 表示"由 u 网络经本构算出"，_s 表示"sigma 网络直接输出"）：
      ux, uy                      位移
      exx, eyy, exy               应变（u 的一阶导）
      sxx_u, syy_u, sxy_u         sigma(u) = A e(u)     <- 只用到 u 的一阶导
      sxx_s, syy_s, sxy_s         sigma^NN              <- 网络直接输出
      en = A e(u):e(u)            能量密度              <- 只用到 u 的一阶导
      div1_s, div2_s              div sigma^NN          <- sigma 网络的一阶导
      div1_u, div2_u              div (A e(u))          <- u 网络的二阶导（Navier 算子）
    残差 r = div + om2*rho*u 在外部按所需 om2 组合（两条路线各组合一次即可对比）。
    """
    xy = XY0.clone().requires_grad_(True)
    if simp is None:
        simp = torch.ones(xy.shape[0], 1, dtype=xy.dtype)
    ux = nets["ux"](xy)
    uy = nets["uy"](xy)
    ux_x, ux_y = _grad(ux, xy)
    uy_x, uy_y = _grad(uy, xy)
    exx, eyy, exy = ux_x, uy_y, 0.5 * (ux_y + uy_x)
    sxx_u, syy_u, sxy_u = stress_from_strain(exx, eyy, exy, cfg.E * simp, cfg.nu)
    en = sxx_u * exx + syy_u * eyy + 2.0 * sxy_u * exy

    # 路线 A：sigma 网络的一阶导
    sxx = nets["sxx"](xy)
    syy = nets["syy"](xy)
    sxy = nets["sxy"](xy)
    sxx_x, _ = _grad(sxx, xy)
    sxy_x, sxy_y = _grad(sxy, xy)
    _, syy_y = _grad(syy, xy)
    div1_s = sxx_x + sxy_y
    div2_s = sxy_x + syy_y

    # 路线 B：u 网络的二阶导（对 sigma(u) 再求一次导 = Navier 算子）
    sxx_u_x, _ = _grad(sxx_u, xy)
    sxy_u_x, sxy_u_y = _grad(sxy_u, xy)
    _, syy_u_y = _grad(syy_u, xy)
    div1_u = sxx_u_x + sxy_u_y
    div2_u = sxy_u_x + syy_u_y

    out = dict(ux=ux, uy=uy, en=en,
               exx=exx, eyy=eyy, exy=exy,
               sxx_u=sxx_u, syy_u=syy_u, sxy_u=sxy_u,
               sxx_s=sxx, syy_s=syy, sxy_s=sxy,
               div1_s=div1_s, div2_s=div2_s,
               div1_u=div1_u, div2_u=div2_u,
               div1=div1_s, div2=div2_s)      # 旧名保留：默认指 sigma 网络那条路线
    return {k: v.detach() for k, v in out.items()}


# ============================== 权重保存 / 载入 ==============================

def save_checkpoint(path, cfg, sdf, nets, extra=None):
    """把 SDF 与 5 个网络的权重、配置、关键标量一起存成一个 .pt。"""
    from dataclasses import asdict
    ckpt = dict(config=asdict(cfg),
                sdf=sdf.state_dict(),
                nets={k: v.state_dict() for k, v in nets.items()},
                extra=dict(extra or {}))
    torch.save(ckpt, path)
    return path


def load_checkpoint(path, cfg=None):
    """载入 checkpoint 并重建网络：返回 (cfg, sdf, nets, extra)。

    用法：
        import torch, pinn_core as core
        torch.set_default_dtype(torch.float64)      # 必须与训练时一致
        cfg, sdf, nets, extra = core.load_checkpoint("outputs/model_final.pt")
        u = nets["ux"](torch.tensor([[0.8, 0.25]]))
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                      # 老版本 torch 没有 weights_only 参数
        ckpt = torch.load(path, map_location="cpu")
    if cfg is None:
        import dataclasses
        from config import Config
        keys = {f.name for f in dataclasses.fields(Config)}   # 容忍配置字段增删
        cfg = Config(**{k: v for k, v in ckpt["config"].items() if k in keys})
    sdf = build_sdf(cfg)
    sdf.load_state_dict(ckpt["sdf"])
    sdf.eval()
    for p in sdf.parameters():
        p.requires_grad_(False)
    nets = build_nets(cfg)
    for k, v in nets.items():
        v.load_state_dict(ckpt["nets"][k])
        v.eval()
    return cfg, sdf, nets, ckpt.get("extra", {})
