"""全局配置：所有开关与超参数集中在这里，改这里即可，不必动其他文件。"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---------- 几何 / 材料 ----------
    Lx: float = 1.6
    Ly: float = 0.5
    E: float = 1.0
    nu: float = 0.3                            # 平面应力
    block: tuple = (0.76, 0.84, 0.21, 0.29)    # 不可优化重块 (x0, x1, y0, y1)
    gamma_block: float = 30.0                  # 重块密度倍数（先 10 试，稳定后可 30 / 100）
    block_smooth_w: float = 0.01               # 重块指示函数 tanh 过渡宽度

    # ---------- SDF / Heaviside ----------
    # Heaviside 带宽。phi=dist(四边) 无偏置时，边界上恒有 S=0.5（与 eps 无关），
    # 但过渡带宽 ~4*eps，收窄它就能把基频拉回满材料。实测（细网格，满材料基准 0.455706）：
    #   eps=0.03 -> 0.346710 (-23.9%)   eps=0.005  -> 0.441392 (-3.1%)
    #   eps=0.01 -> 0.416656 ( -8.6%)   eps=0.0025 -> 0.453595 (-0.5%)
    #                                   eps=0.001  -> 0.455700 (-0.0%)
    # 下限由三件事卡着：细网格单元高 0.01、粗网格 0.02、SDF 网络的拟合误差(~1e-3)。
    # eps 掉到与拟合误差同量级时，边界的 S 就由网络噪声决定了，诊断里会告警。
    eps_heaviside: float = 0.0025              # S = 0.5*(1 + tanh(phi/eps))
    # SDF 初始化方式：
    #   "const"     phi ≡ sdf_target，全域满材料。|grad phi|=0，不是真正的距离函数，
    #               但本阶段（与 FEM 对拍满材料板）最干净，是默认值。
    #   "edge_dist" phi = dist(x, 四边) + sdf_offset，真正的距离函数（|grad phi|=1）。
    #               注意 sdf_offset=0 时边界上 phi=0 => S=0.5，夹持边变成"半材料"，
    #               实测基频掉 24%，而且 Dirichlet/Neumann 损失被 S^2 乘成 0.25 倍。
    #               取 sdf_offset >= 5*eps 就既是距离函数、边界又是满材料。
    #   "holes"     phi = min(dist(四边)+offset, 各播种圆孔的符号距离)，
    #               水平集拓扑优化的经典初值（水平集法在 2D 无法自发生成新孔）。
    sdf_init: str = "edge_dist"
    sdf_target: float = 0.2                    # const 模式下 phi 的常数值
    sdf_offset: float = 0.0                    # 非 const 模式下的整体上抬；
                                               # 0 = 零水平集压在域边界上（边界 S=0.5，靠收窄 eps 补偿）
                                               # 5*eps = 零水平集推到域外（边界 S=1，与 eps 解耦）
    sdf_holes: tuple = ((0.40, 0.25, 0.09), (1.20, 0.25, 0.09))   # 播种孔 (x, y, r)
    sdf_pretrain_iters: int = 5000             # const 用 800 够；edge_dist / holes 需要 5000
    sdf_pretrain_lr: float = 2e-3
    sdf_plot_nx: int = 321                     # SDF / 材料场诊断图的绘图网格
    sdf_plot_ny: int = 101

    # ---------- SDF 预训练强化：边界带加采样 + L-BFGS 微调 ----------
    # S = H(phi/eps) 只对 |phi| <~ 4*eps 的窄带敏感；域四边与孔缘恰好落在带里，
    # 那里的拟合误差直接变成 S 的失真（err=1e-3、eps=0.0025、offset=0 时，
    # 边界 S 会从 0.5 漂到 ~0.69），再经掩码进入 Dirichlet/Neumann 损失与 FEM 密度场。
    # 因此预训练在均匀点之外对这条带单独采样、单独加权（Adam 阶段），
    # 再用固定点集上的全批量 L-BFGS 把带内误差压低两到三个量级（微调阶段）。
    sdf_bnd_band: float = 0.02          # 边界带宽 ~= 8*eps，覆盖两个 tanh 过渡带
    sdf_batch_bnd: int = 1024           # Adam 每步的边界带点数（0 = 关闭加采样，回到旧行为）
    sdf_w_bnd: float = 10.0             # 边界带 MSE 权重（损失 = mse_int + w_bnd*mse_bnd）
    sdf_lbfgs_outer: int = 20            # L-BFGS 分块数（0 = 关闭微调）；总迭代 = outer*per
    sdf_lbfgs_max_iter_per: int = 70    # 每块内部迭代上限
    sdf_lbfgs_int: int = 16384          # 固定内部点数（Sobol；全批量保证损失确定性）
    sdf_lbfgs_bnd: int = 4096           # 固定边界带点数
    sdf_lbfgs_stop_rtol: float = 1e-3   # 一块的相对改善低于它算作停滞
    sdf_lbfgs_patience: int = 2         # 连续停滞多少块就早停（结束回滚到验证最优）

    # ---------- FEM 基准 ----------
    nx_fem: int = 160
    ny_fem: int = 50
    nx_fem_coarse: int = 80
    ny_fem_coarse: int = 25
    n_modes_fem: int = 6
    # 刚度是否按 S^simp_p 缩放。关键：这个开关同时作用于 FEM 与 PINN，
    # 保证两边永远是同一个算子——否则 S 一旦不恒为 1（比如 edge_dist 初始化），
    # FEM 用 S^3、PINN 用 1，边界处差 8 倍，基频能差 24%，对拍就没意义了。
    # False = 按你 md 里的写法：A 是基础弹性张量，S 只当掩码用（当前阶段）
    # True  = SIMP 插值，进拓扑优化时打开，两边同时生效
    use_simp: bool = False
    simp_p: int = 3          # use_simp=True 时的指数
    mode_index: int = 0      # 目标模态：0 = 最小 omega^2（无监督地锁定基频）
    use_richardson_target: bool = False   # True: 用 Richardson 外推值 (4*lam_f-lam_c)/3 当训练锚点

    # ---------- 网络 ----------
    hidden: int = 64
    layers: int = 3
    fourier: bool = False    # Fourier 特征嵌入开关（默认关；场偏陡时可开）
    n_ff: int = 32
    sigma_ff: float = 2.0

    # ---------- 采样（CPU 友好规模）----------
    n_pool_int: int = 20000      # 内部 Sobol 点池
    n_pool_blk: int = 3000       # 重块及邻域加密点池
    n_pool_bnd: int = 2000       # 每条边的边界点池
    batch_int: int = 4096
    batch_blk: int = 512
    batch_bnd: int = 256         # 每条边每步取的点数
    block_sample_margin: float = 0.02   # 重块采样外扩边距（覆盖过渡带）

    # ---------- 训练 ----------
    adam_iters: int = 4000
    lr: float = 1e-3
    lr_min: float = 1e-5         # 余弦退火终点
    rayleigh_warmup: int = 500 #500   # 前 warmup 步不加 Rayleigh 损失（随机初值下 R 很大，防止梯度爆冲）
    lbfgs_outer: int = 60         # L-BFGS 分块数（每块之间保存快照）；总迭代 = outer*per
    lbfgs_max_iter_per: int = 70 # 每块内部迭代上限，总计 outer*per
    lbfgs_int: int = 8000        # L-BFGS 固定点集规模（必须固定，保证损失确定性）
    lbfgs_blk: int = 800
    lbfgs_bnd: int = 300
    lbfgs_val_int: int = 6000     # 独立验证点集（只评估不训练，用来发现"过拟合配点"）
    lbfgs_stop_rtol: float = 1e-4 # 一块的相对改善低于它算作停滞
    lbfgs_patience: int = 4       # 连续停滞多少块就早停（省时间，跑不满 outer 也没关系）
    lbfgs_restore_best: bool = True  # 结束时回到验证损失最优的那一块

    # ---------- 材料掩码指数 ----------
    # 残差型损失（pde / cons / dir / neu）前乘 S^mask_pow_loss；
    # Rayleigh 商的分子分母（含质量归一化）用 S^mask_pow_ray。
    # 约定：损失里用 S 的一次方，Rayleigh 商里用 S^2。
    # 注意 phi=dist 无偏置时边界上 S 恒为 0.5，等于把 w_dir / w_neu 打了对折，
    # 想抵消就把这两个权重乘 2。
    mask_pow_loss: float = 1.0
    mask_pow_ray: float = 2.0

    # ---------- 损失权重（各项已无量纲化，固定权重即可"一次跑对"）----------
    w_pde: float = 1.0
    w_cons: float = 1.0          # 本构残差（把 sigma 网络与 u 网络绑定）
    w_dir: float = 10.0          # 左右 Dirichlet
    w_neu: float = 1.0           # 上下 Neumann（sigma_xy = sigma_yy = 0）
    w_norm: float = 10.0         # 质量归一化 (int S^2 rho |u|^2 = 1)，排除 u=0 平凡解
    w_ray: float = 1.0           # (R/omega^2_FEM - 1)^2
    w_blk: float = 1.0           # 重块加密点上的 pde+cons 附加项
    adaptive_weights: bool = False   # 梯度范数自适应加权开关（默认关，保确定性）
    adapt_every: int = 250
    adapt_alpha: float = 0.1
    hard_normalization: bool = False # 硬归一化开关：投影式缩放代替软约束（默认关）
    omega2_trainable: bool = False   # omega^2 是否可训练（默认固定为 FEM 值）

    # ---------- 输出 / 其他 ----------
    snapshot_every: int = 250    # 每多少步保存一帧 ux/uy 演变图
    log_every: int = 100
    save_ckpt: bool = True       # 保存 sdf + 5 个网络的权重（.pt）
    seed: int = 0
    outdir: str = "outputs"

    @property
    def area(self):
        return self.Lx * self.Ly
