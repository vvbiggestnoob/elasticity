# HJB 训练代码使用文档

> 本文档对应 `start/hjb_step.py` 的 PyTorch 实现，规格的唯一权威说明见工作区根目录
> 《HJB方程优化说明.md》，数值取自《实验设置与计算范围.md》。修改实现时文档需同步修订。
>
> **2026-09-08 方法修订**：移除外边界损失 L_b（参考论文对 SDF 无域边界条件）；
> 初始条件增加边界带加密点；新增制造解锚点 L_anchor 破除 HJB 残差的平解退化。
> 详见 §5.3。

---

## 1. 文件清单

```
start/
├── networks.py          # 网络创建/保存/加载（复用，不修改）
├── mech_init.py         # 上一步：力学初始化（产出 weights/u_x_init.pt 等 5 个权重）
├── sdf_init.py          # 上一步：SDF 初始化（产出 weights/phi_init.pt）
├── fem_solver.py        # FEM 求解器（本步骤诊断 §9 的演化后 ω²_FEM 对拍复用）
├── hjb_step.py          # 本步骤：单次 HJB 演化（PINN，两阶段训练，推进一个 dt）
├── hjb_fem.py           # 对照基准：同一初值问题的网格（Godunov 迎风）求解
└── （运行后生成）
    ├── hjb_figures/     # V_n 场图 + 训练过程图片 + 损失曲线 + 最终结果图
    └── weights/
        ├── phi_init.pt      # （输入）当前 SDF（t_current = 0 切片已拟合）
        ├── u_x_init.pt 等   # （输入）5 个冻结力学网络
        ├── phi_iter1.pt     # （输出）演化后 SDF 网络权重（纯 state_dict）
        └── hjb_state.json   # （输出）伪时间 t_new、F/ω²/面积前后值、诊断记录
```

依赖：Python 3 + PyTorch（float64，CPU 可跑）+ matplotlib（Agg 后端只存图）；
FEM 对拍另需 scipy（`fem_solver.py` 的既有依赖，可用 `fem_check=False` 关闭）。
运行前需已有 `weights/phi_init.pt` 与 5 个 `weights/*_init.pt` 力学网络权重。

## 2. 实现内容（与《HJB方程优化说明.md》的对应关系）

| 说明文档规格 | 实现 |
|---|---|
| 冻结力学网络，加载 5 个 `*_init.pt`（§2） | `load_frozen_mech()`（`requires_grad_(False)`） |
| SDF 双份：冻结锚点 + 可训练热启动 | `load_phi(frozen=True/False)` → `phi_ref` / `phi` |
| V_n = ε:A:ε − ω²ρ‖u‖² − (1/α)(∫S dx − C)，算一次后固定（§2、§5） | `compute_vn()`（逐点预计算常数，随点集存储） |
| ε:A:ε 由本构 σ^u:ε 给出（与 Rayleigh 商约定一致） | `frozen_strain()` + `constitutive_stress()` |
| ω² 由 Rayleigh 商给出（§3） | `rayleigh_and_area()`（65536 Sobol 积分点集） |
| α = 1.0 固定（§2） | `HJBConfig.alpha` |
| (1) HJB 残差 r = ∂φ/∂t − V_n·\|∇φ\|（§7，符号见 §5.1） | `hjb_residual_loss()` |
| (2) 初始条件，含边界带加密（§7） | `init_loss()`（逐点权重：内部 1 / 边界带 `w_ic_band=10`） |
| (3) 制造解锚点（§7，可选，λ_anchor=1） | `anchor_loss()` + `phi_ref_val_and_gradnorm()` |
| ~~外边界零法向梯度 L_b~~ | **已于 2026-09-08 移除**（见 §5.3） |
| 不乘掩码 S（§7） | 三项损失均不乘 S |
| 阶段 A：Sobol 点池 + 每步 minibatch 重采，余弦退火（§8） | `train_adam()` |
| 阶段 B：L-BFGS 固定点集全批量，强 Wolfe，优化器跨块复用（§8） | `train_lbfgs()`（与 `mech_init.py` 同策略） |
| 早停 / 回滚默认关闭，验证集只打印观察（§8） | `cfg.lbfgs_early_stop` / `cfg.lbfgs_rollback`，置 True 开启 |
| 诊断 §9：残差 RMS、F/ω²/面积前后、零水平集对比、\|∇φ\|、FEM 对拍 | `diagnose()` + `grad_norm_near_zero_level()` + `fem_cross_check()` |
| 保存 phi_iter1.pt 与 t_current+dt（§10） | `weights/phi_iter1.pt` + `weights/hjb_state.json` |
| 位移/应力/应变乘 S 后出图（§10） | `save_mech_fields_masked()` |

采样规模（《实验设置与计算范围.md》§11）：Adam 点池 20000（时空 3D Sobol）+
10000（初始条件 2D Sobol）+ 4000（IC 边界带）；每步 batch 2000 + 1000（合并池
抽取）；L-BFGS 固定 8192 + 4096 + 1024(带)；验证集 4096 + 2048 + 512(带)。

## 3. 快速开始

```bash
cd start
python hjb_step.py
```

> 路径说明：所有相对路径相对 `hjb_step.py` 脚本所在目录解析（`SCRIPT_DIR`），
> 从任意工作目录运行（如工作区根目录 `python start/hjb_step.py`）均可。

全程 CPU 可跑（float64，分钟级）。运行过程：

1. **准备**：加载冻结力学网络与 SDF（双份）；在 65536 Sobol 积分点集上算
   ω²（Rayleigh 商）与面积 ∫S dx；构造 V_n 场并打印统计（mean/min/max），
   保存 `vn_field.png`。
2. **阶段 A（Adam）**：每步从点池重采 minibatch。每 `log_every` 步打印
   （total / r / init / anchor / lr），每 `fig_every` 步存一张四联图。
3. **阶段 B（L-BFGS）**：固定点集全批量，优化器跨块复用，每块最多 70 次内部
   迭代，**按块**打印（train / val / 相对改善），每 5 块存图（末块必存）。
   默认不早停、不回滚。
4. **收尾**：保存 `loss_history.png`、`final.png`、`mech_fields_masked.png`、
   `weights/phi_iter1.pt`、`weights/hjb_state.json`，打印诊断结果；
   `fem_check=True`（默认）时对演化后几何跑 160×50 细网格 FEM 对拍。

自定义配置（如冒烟测试）：

```python
from hjb_step import HJBConfig, main

cfg = HJBConfig(adam_steps=50, lbfgs_blocks=2, fem_check=False,
                fig_dir="hjb_figures_test", phi_out="weights_test/phi_iter1.pt",
                state_out="weights_test/hjb_state.json")
main(cfg)
```

关键开关：

| 配置 | 默认 | 说明 |
|---|---|---|
| `lam_anchor` | 1.0 | 制造解锚点权重；**置 0 退回纯 PINN**（L_r + L_0 两项） |
| `w_ic_band` | 10.0 | 初始条件边界带点权重（同 sdf_init 的 w_bnd） |
| `pool_init_band` | 4000 | IC 边界带点数（带宽 4β，隔点严格压边） |
| `pool_init_corner` | 1000 | IC 角点加密（4β 见方角域，与带同权重；L-BFGS/验证集为 256/128） |
| `fem_check` | True | 演化后几何的 FEM ω² 对拍 |

## 4. 输出解读

### 4.1 打印日志

- **V_n 统计**：mean ≈ −(∫S dx − C)/α ≈ −0.37（全材料阶段罚项主导）；
  重块处因 ω²ρ‖u‖² 大出现强负峰（约 −25，去材料趋势）；固支端角点附近
  应变能集中，V_n 为正（约 +1，保留材料趋势）。
- **r 列**（HJB 残差）应持续下降；**init 列**（初始条件）从 0 升起后应压回
  小量；**anchor 列**（制造解锚点，λ_anchor>0 时）应压到与 r 同量级或更小。
- **诊断**：F 应下降（V_n 取最速下降方向）；面积应向 C = 0.4 靠近
  （单步 dt 变化很小）；FEM 对拍给出演化后几何的 ω²_FEM 与 PINN Rayleigh 商对比。

### 4.2 hjb_state.json

记录 `t_current` / `dt` / `t_new`、`alpha`、`v_target`、演化前后的
面积 / ω²（PINN Rayleigh）/ F、HJB 残差 RMS（train/val）、零水平集附近
|∇φ| 统计、FEM 对拍结果、随机种子与耗时。下一次外层迭代以此为输入状态。

### 4.3 图片

`hjb_figures/` 下：`vn_field.png`（V_n 及其应变能/动能分量三联图）、
`step000000_initial.png`、`adam_step*.png`、`lbfgs_block*.png`（四联图：
φ(t_new) + 零水平集前后对比、S 演化前、S 演化后、ΔS）、
`loss_history.png`（左 Adam 各分量、右 L-BFGS train/val）、
`final.png`、`mech_fields_masked.png`（位移/应力/应变 × S(t_new)，孔洞置 0）。

## 5. 关键设计说明（实现层面的决策记录）

### 5.1 HJB 残差符号（重要）

说明文档 §4 给出 ∂φ/∂t + V_n|∇φ| = 0，§5 给出 V_n 公式并明确
"V_n > 0 边界外扩（加材料）"。二者在"φ > 0 = 材料"的约定下**符号冲突**：
∇φ 在边界处指向材料内部，按 §4 方程 V_n > 0 会使边界向材料内部运动（收缩），
与 §5 的物理描述和最速下降推导（dF = −∫G² < 0）矛盾。参考论文
（DNN Level Set TO，式 (5)）的方程为 **∂Φ/∂t − V_n|∇Φ| = 0**，与 §5 自洽。

实现采用与论文及 §5 一致的形式：**r = ∂φ/∂t − V_n·|∇φ|**，V_n 仍按 §5 公式
计算（V_n > 0 → 边界外扩）。说明文档 §4/§7/§11 已补勘误记录。

### 5.2 V_n 的预计算与"算一次后固定"

V_n 依赖 ω² 与面积（两个全局积分量，用 65536 Sobol 积分点集估计一次）和逐点的
ε:A:ε、ρ‖u‖²（冻结力学网络一阶自动微分）。Adam 点池 / L-BFGS 固定集 /
验证集在构建时各自预计算 V_n 存为常数张量（`InteriorSet.vn`），训练中查表
使用，整个演化过程 V_n 不变（说明文档 §2）。

### 5.3 边界处理与 rim 虚增问题（2026-09-08 方法修订，重要）

**L_b 已移除。** 原实现有第三项损失 L_b（外边界零法向梯度）。参考论文对 SDF
不施加域边界条件；且 L_b 与 L_0 在 t=0 棱线上冲突（初始 SDF 在边界上
|∂φ/∂n| = 1 ≠ 0）；网格对照版还证明零 Neumann 条件会把边界值钉死
（φ_t = 0），零水平集永远无法从域边界后退。移除后边界为出流式
（值由 PDE + 内部信息决定），边界可自由后退。

**rim 虚增的真正主因是初始条件锚定不足（不是 L_b）。** 移除 L_b 前后 rim 处
φ 虚增量几乎相同（+0.077 vs +0.076，面积均 +0.028）——实测证实 L_b 并非主因。
真正机制：**HJB 残差存在平解退化**——常函数 φ ≡ const 的残差恒为零
（φ_t = 0、|∇φ| = 0），残差只约束比值 φ_t/|∇φ| = V_n，φ 的**值**完全靠
初始条件钉。而纯内部 Sobol 采样下 rim 是零测集、永远采不到点，光滑网络把
边界拐角（φ = 0 的最小值正在边界上）抹圆，rim 处 φ 系统性虚增 → S 由 0.5
抬到 ~1 → 面积虚增。

**两道修复**：

1. **IC 边界带**（`sample_init_band()`）：初始条件点中加入边界带加密点
   （带宽 4β = 0.04、隔点严格压在边上、外加 4 角点，权重 `w_ic_band = 10`），
   与 `sdf_init.py` 的边界带同策略，直接把 rim 钉在 φ_ref 上。
   另有**角点加密**（`sample_init_corners()`）：四个角各在 4β 见方的角域内
   均匀加密（角域是两条边界带的交汇区、rim 曲率最大处，最容易被抹圆），
   角点并入边界带、同权重 `w_ic_band`。
2. **制造解锚点 L_anchor**（`anchor_loss()`，λ_anchor = 1，置 0 关闭）：
   锚定 φ 到一阶制造解 φ_ref + (t−t₀)·V_n·|∇φ_ref|（`phi_ref_val_and_gradnorm()`
   预计算 |∇φ_ref|）。给出整个时间柱面上的显式剖面目标，从根上破除平解退化。
   dt 小（0.005）时一阶制造解与真解几乎一致；它与网格对照版（hjb_fem.py）
   的 Godunov 推进在一阶意义下等价。

> 注意：制造解锚点为本次新增，代码经静态审查但**尚未经完整训练验证**；
> 若训练行为异常，先置 `lam_anchor=0` 退回纯 PINN（L_r + L_0 + IC 边界带）
> 再排查。

### 5.4 其他

1. **"演化后 ω²" 是冻结力学近似**：诊断中演化后的 ω² 用新几何的 S 场 +
   冻结力学网络重算 Rayleigh 商得到，并非重新求解特征值问题；真正的 ω²
   更新属于下一次外层迭代的力学重解。FEM 对拍（`fem_check`）给出独立基准。
2. **t 通道不归一化**：t ∈ [0, dt] 量值很小，网络只需在 φ_ref 基础上
   叠加 O(dt) 的修正（∂φ/∂t ~ O(V_n)），与 SDF 初始化阶段的约定一致。
3. **验证/诊断不开二阶图**：`create_graph=False` 只算一阶导；训练路径恒为
   `create_graph=True`。
4. **|∇φ| 加 1e-30 保护**：`sqrt(g²+1e-30)` 避免零梯度处 backward 出 NaN。
5. **等值线图例用 Line2D 代理**：零水平集可能不在图域内（空等值线），
   直接从 contour collection 取 handle 会崩。
6. **phi_iter1.pt 为纯 state_dict**：与 `networks.py` 的加载约定一致，
   可直接 `load_state_dict`；伪时间等元数据在 `hjb_state.json`。

## 6. 后续衔接

本步骤产物供外层交替迭代使用：

- `weights/phi_iter1.pt`：下一轮的新当前 SDF。下一轮 HJB 时
  `t_current = 0.005`（`hjb_state.json` 的 `t_new`），初始条件锚点取
  该网络在 `t_current` 切片的输出；
- 外层循环下一步是力学网络在新几何上的重解（Step A），随后回本步骤
  （Step B）；罚参数自适应更新（Step D，说明文档 §6）与周期性重初始化
  （《重初始化方案说明》，多步推进时启用）在外层循环阶段接入。

加载演化后 SDF（与 `networks.py` 约定一致）：

```python
import torch
from networks import NetworkConfig, build_sdf_network

phi = build_sdf_network(NetworkConfig())
phi.load_state_dict(torch.load("weights/phi_iter1.pt", weights_only=True))
# 评估 t = 0.005 切片：xyt = torch.cat([xy, torch.full((N,1), 0.005)], dim=1)
```
