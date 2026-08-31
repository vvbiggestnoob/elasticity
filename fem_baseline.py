"""纯 NumPy/SciPy 的 Q4 平面应力广义特征值 FEM 基准。

两种用法：
1. 被 train.py 调用：单元密度/刚度由（冻结的）SDF 网络给出，得到 ω²_FEM 与参考模态；
2. 独立运行 `python fem_baseline.py`：用解析密度（满材料 + 重块）打印基准特征值，
   用于健全性检查与网格收敛性评估。

约定：
- 位移场 (ux, uy)，节点编号 n = j*(nx+1)+i（i 沿 x 方向最快），自由度 (2n, 2n+1)；
- 左右边界 (x=0, x=Lx) 全固支 Dirichlet；上下边界自然满足零面力 Neumann；
- 一致质量阵；广义特征值 K φ = ω² M φ 用 shift-invert eigsh 求最小的若干个。
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def plane_stress_C(E, nu):
    """平面应力本构矩阵（Voigt 记法，作用于 (exx, eyy, gamma_xy)）。"""
    c = E / (1.0 - nu ** 2)
    return c * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])


def q4_shapes(xi, eta, hx, hy):
    """Q4 矩形单元在参考坐标 (xi, eta) 处的形函数矩阵。

    返回 (N, B, Nm)：
      N  : (4,)  形函数值
      B  : (3,8) 应变-位移矩阵，作用在 (ux1,uy1,...,ux4,uy4) 上给出 (exx, eyy, gamma_xy)
      Nm : (2,8) 位移插值矩阵
    K、M 与后处理的 q 场全部走这一个函数，保证用的是同一套形函数。
    """
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    dN_dx = dN_dxi * (2.0 / hx)
    dN_dy = dN_deta * (2.0 / hy)
    B = np.zeros((3, 8))
    Nm = np.zeros((2, 8))
    for i in range(4):
        B[0, 2 * i] = dN_dx[i]
        B[1, 2 * i + 1] = dN_dy[i]
        B[2, 2 * i] = dN_dy[i]
        B[2, 2 * i + 1] = dN_dx[i]
        Nm[0, 2 * i] = N[i]
        Nm[1, 2 * i + 1] = N[i]
    return N, B, Nm


def q4_element_matrices(hx, hy, C):
    """单位刚度/单位密度下的 Q4 矩形单元 Ke, Me（2x2 Gauss 积分）。"""
    g = 1.0 / np.sqrt(3.0)
    Ke = np.zeros((8, 8))
    Me = np.zeros((8, 8))
    detJ = hx * hy / 4.0
    for xi in (-g, g):
        for eta in (-g, g):
            _, B, Nm = q4_shapes(xi, eta, hx, hy)
            Ke += B.T @ C @ B * detJ
            Me += Nm.T @ Nm * detJ
    return Ke, Me


def element_dofs(nx, ny):
    """单元 -> 8 个自由度的节点编号 (n1, n2, n3, n4)，逆时针，n = j*(nx+1)+i。"""
    ie = np.tile(np.arange(nx), ny)
    je = np.repeat(np.arange(ny), nx)
    n1 = je * (nx + 1) + ie
    n2 = n1 + 1
    n3 = n2 + (nx + 1)
    n4 = n1 + (nx + 1)
    return ie, je, (n1, n2, n3, n4)


def mode_q_fields(nx, ny, Lx, Ly, E, nu, ux_nodes, uy_nodes, dens_fn, stiff_fn, lam):
    """把 FEM 模态代回单元，算 q = s*Ae(u):e(u) - lam*rho*|u|^2。

    与 PINN 那边同一个公式，只是 u 换成 FEM 的有限元解 u_h（双线性形函数），
    导数由 B 矩阵给出（逐单元精确，不做差分）。

    返回 dict：
      xc, yc, en, kin, q, rho, ux, uy   —— 单元中心值，形状 (ny, nx)，直接可画
      I_en, I_kin, I_q                  —— 2x2 Gauss 精确积分
    因为 I_en = phi^T K phi、I_kin = phi^T M phi，对精确本征对必有 I_q = I_en - lam*I_kin = 0
    （机器精度），这就是这段代码自带的校核。
    """
    hx, hy = Lx / nx, Ly / ny
    C = plane_stress_C(E, nu)
    ie, je, nodes = element_dofs(nx, ny)
    xc = (ie + 0.5) * hx
    yc = (je + 0.5) * hy
    rho_e = np.asarray(dens_fn(xc, yc), dtype=float)
    s_e = np.asarray(stiff_fn(xc, yc), dtype=float)

    uxf = np.asarray(ux_nodes, dtype=float).ravel()
    uyf = np.asarray(uy_nodes, dtype=float).ravel()
    ue = np.empty((nx * ny, 8))
    for k, nd in enumerate(nodes):
        ue[:, 2 * k] = uxf[nd]
        ue[:, 2 * k + 1] = uyf[nd]

    # ---- 2x2 Gauss：精确积分（权重都是 1）----
    g = 1.0 / np.sqrt(3.0)
    detJ = hx * hy / 4.0
    I_en = 0.0
    I_kin = 0.0
    for xi in (-g, g):
        for eta in (-g, g):
            _, B, Nm = q4_shapes(xi, eta, hx, hy)
            eps = ue @ B.T                       # (Ne,3) = (exx, eyy, gamma_xy)
            sig = eps @ C.T                      # (Ne,3) = (sxx, syy, sxy)
            en_g = np.einsum("ij,ij->i", sig, eps)   # Voigt 点积已含 2*sxy*exy
            u_g = ue @ Nm.T                      # (Ne,2)
            I_en += detJ * float(np.sum(s_e * en_g))
            I_kin += detJ * float(np.sum(rho_e * (u_g ** 2).sum(axis=1)))

    # ---- 单元中心值：用于画图 ----
    _, B0, Nm0 = q4_shapes(0.0, 0.0, hx, hy)
    eps0 = ue @ B0.T
    sig0 = eps0 @ C.T
    en_c = s_e * np.einsum("ij,ij->i", sig0, eps0)
    u_c = ue @ Nm0.T
    kin_c = rho_e * (u_c ** 2).sum(axis=1)
    shp = (ny, nx)
    return dict(xc=xc.reshape(shp), yc=yc.reshape(shp),
                en=en_c.reshape(shp), kin=kin_c.reshape(shp),
                q=(en_c - lam * kin_c).reshape(shp),
                rho=rho_e.reshape(shp),
                ux=u_c[:, 0].reshape(shp), uy=u_c[:, 1].reshape(shp),
                I_en=I_en, I_kin=I_kin, I_q=I_en - lam * I_kin)


def solve_plane_stress_eig(nx, ny, Lx, Ly, E, nu, dens_fn, stiff_fn, n_modes=6):
    """求解广义特征值问题 K φ = ω² M φ。

    dens_fn / stiff_fn: 输入单元中心坐标一维数组 (xc, yc)，返回每单元密度 / 刚度缩放。
    返回 dict(lams, modes_ux[k, ny+1, nx+1], modes_uy[...], xs, ys)，模态已做 M 归一化。
    """
    hx, hy = Lx / nx, Ly / ny
    C = plane_stress_C(E, nu)
    Ke0, Me0 = q4_element_matrices(hx, hy, C)

    Ne = nx * ny
    ie, je, (n1, n2, n3, n4) = element_dofs(nx, ny)
    xc = (ie + 0.5) * hx
    yc = (je + 0.5) * hy
    rho_e = np.asarray(dens_fn(xc, yc), dtype=float)
    s_e = np.asarray(stiff_fn(xc, yc), dtype=float)

    edof = np.stack([2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
                     2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1], axis=1)  # (Ne, 8)

    rows = np.broadcast_to(edof[:, :, None], (Ne, 8, 8)).ravel()
    cols = np.broadcast_to(edof[:, None, :], (Ne, 8, 8)).ravel()
    nd = 2 * (nx + 1) * (ny + 1)
    K = sp.coo_matrix(((s_e[:, None, None] * Ke0[None, :, :]).ravel(), (rows, cols)),
                      shape=(nd, nd)).tocsr()
    M = sp.coo_matrix(((rho_e[:, None, None] * Me0[None, :, :]).ravel(), (rows, cols)),
                      shape=(nd, nd)).tocsr()
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)

    # 左右边界全固支
    jj = np.arange(ny + 1)
    fixed_nodes = np.concatenate([jj * (nx + 1) + 0, jj * (nx + 1) + nx])
    fixed = np.sort(np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1]))
    free = np.setdiff1d(np.arange(nd), fixed)
    Kff = K[free][:, free].tocsc()
    Mff = M[free][:, free].tocsc()

    vals, vecs = spla.eigsh(Kff, k=n_modes, M=Mff, sigma=0.0, which="LM")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    modes_ux, modes_uy = [], []
    for k in range(n_modes):
        v = vecs[:, k]
        v = v / np.sqrt(v @ (Mff @ v))       # M 归一化
        full = np.zeros(nd)
        full[free] = v
        modes_ux.append(full[0::2].reshape(ny + 1, nx + 1))
        modes_uy.append(full[1::2].reshape(ny + 1, nx + 1))
    return dict(lams=vals, modes_ux=np.array(modes_ux), modes_uy=np.array(modes_uy),
                xs=xs, ys=ys)


def analytic_chi_block(x, y, block, w):
    """重块的光滑指示函数（NumPy 版，与 pinn_core 中 torch 版一致）。"""
    bx = 0.5 * (np.tanh((x - block[0]) / w) + np.tanh((block[1] - x) / w))
    by = 0.5 * (np.tanh((y - block[2]) / w) + np.tanh((block[3] - y) / w))
    return bx * by


def richardson(lam_coarse, lam_fine, order=2, ratio=2):
    """网格加密比 ratio、收敛阶 order 时的 Richardson 外推：更接近连续解的特征值。

    Q4 位移元 + 一致质量阵的特征值从上方收敛且 lam_h = lam + C h^order，
    ratio=2, order=2 时即 (4*lam_fine - lam_coarse)/3。
    """
    f = ratio ** order
    return (f * lam_fine - lam_coarse) / (f - 1.0)


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    Lx, Ly, E, nu = cfg.Lx, cfg.Ly, cfg.E, cfg.nu
    block = cfg.block
    gamma, w = cfg.gamma_block, cfg.block_smooth_w

    def dens(xc, yc):        # 满材料 S=1，重块处密度 gamma
        return 1.0 + (gamma - 1.0) * analytic_chi_block(xc, yc, block, w)

    def stif(xc, yc):
        return np.ones_like(xc)

    def dens_uniform(xc, yc):
        return np.ones_like(xc)

    lams = {}
    for nx, ny, tag in [(cfg.nx_fem_coarse, cfg.ny_fem_coarse, "粗"),
                        (cfg.nx_fem, cfg.ny_fem, "细")]:
        r = solve_plane_stress_eig(nx, ny, Lx, Ly, E, nu, dens, stif, cfg.n_modes_fem)
        lams[tag] = r["lams"]
        print(f"[满材料+重块 gamma={gamma:g}] {tag} {nx}x{ny}: omega^2 =",
              np.array2string(r["lams"], precision=6))
    print("[Richardson 外推 (4*细-粗)/3]              omega^2 =",
          np.array2string(richardson(lams["粗"], lams["细"]), precision=6))

    r_u = solve_plane_stress_eig(cfg.nx_fem, cfg.ny_fem, Lx, Ly, E, nu,
                                 dens_uniform, stif, cfg.n_modes_fem)
    print("[满材料 无重块]                            omega^2 =",
          np.array2string(r_u["lams"], precision=6))
