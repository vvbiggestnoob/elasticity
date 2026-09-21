# FEM 求解器代码使用文档

> 本文档对应 `start/fem_solver.py` 的实现，规格的唯一权威说明见工作区根目录
> 《FEM计算说明.md》，数值取自《实验设置与计算范围.md》。修改实现时文档需同步修订。

---

## 1. 文件清单

```
start/
├── networks.py          # 网络创建/保存/加载（复用，不修改）
├── sdf_init.py          # SDF 初始化（产出 weights/phi_init.pt）
├── mech_init.py         # 应力/位移初始化（消费 FEM 锚点 omega2_fem）
├── fem_solver.py        # 本步骤：FEM 特征值基准求解（给定 SDF）
└── （运行后生成）
    ├── fem_figures/     # fem_fields_mode1.png（5 个场 + 材料场）+ fem_eigenvalues.png
    └── fem_results/     # fem_result.npz（特征值 + 参考模态 + 网格）+ fem_summary.txt
```

依赖：Python 3 + numpy + scipy（sparse / eigsh）+ matplotlib（Agg 后端，只存图）
+ PyTorch（仅用于 SDF 网络前向取值）。已在 torch 2.4.1+cpu / scipy 1.10.1 /
numpy 1.24.4 上核对。全程 float64，CPU 秒级完成。

运行前需已有 SDF 网络权重（默认 `weights/phi_init.pt`，SDF 初始化产物）。

## 2. 实现内容（与《FEM计算说明.md》的对应关系）

| 说明文档规格 | 实现 |
|---|---|
| S = H(φ) = ½(1+tanh(φ/2β))，β=0.01，单元中心采样（§2） | `load_sdf_fn()` + `element_materials()` |
| ρ_e = S_e·ρ̂_e（空洞无质量）；ρ̂：重块 100、其余 1（§2） | `element_materials()`（重块硬指示函数，与 `mech_init.py` 约定一致） |
| E_e = S_e·E_solid + (1−S_e)·E_void，E_void=1e-6 ersatz（§2） | `element_materials()` |
| Q4 双线性单元，2×2 高斯积分，平面应力 ν=0.3（§3） | `q4_template()` / `_q4_B()` / `plane_stress_C0()` |
| 一致质量阵 M_e = ρ_e∫NᵀN（§3） | `q4_template()` 的 `M0` |
| 细网格 160×50、粗网格 80×25，hx=hy=0.01（§6） | `FEMConfig.nx_fine/ny_fine/nx_coarse/ny_coarse` |
| 左、右边界全固支（消去法）；上、下自由（§4） | `clamped_free_dofs()` |
| shift-invert `eigsh`，sigma=0，取最小若干阶（§4） | `solve_eigen()`（`scipy.sparse.linalg.eigsh`，k=`n_eig`=6） |
| 模态质量归一化 φᵀMφ=1（§4） | `solve_eigen()` 末尾显式归一化 + 符号约定（最大绝对值分量为正） |
| Richardson 外推 (4·ω²_细 − ω²_粗)/3（§6） | `main()` 内 |
| 画图乘 H(SDF) 过滤空洞，叠加 φ=0 零等值线（§5） | `plot_fields()`（位移用节点 S，应力用单元中心 S） |
| 输出 ω²_FEM 标量 + 参考模态 + 过滤场图（§7） | `fem_result.npz` + `fem_summary.txt` + `fem_figures/` |

实现要点：均匀矩形网格下所有单元的"单位模板"相同，`q4_template()` 只算一次
K0（E=1）/ M0（ρ=1），逐单元乘 E_e / ρ_e 后用 COO 一次性组装（`assemble()`）。

## 3. 快速开始

```bash
cd start
python fem_solver.py
```

全程 CPU 秒级（细网格 16422 自由度的 shift-invert 约 1s）。运行过程：

1. 加载 SDF 网络（t=0 切片），在细网格 160×50 单元中心采样 φ → S、ρ、E；
2. 组装 K、M，消去左右固支自由度，shift-invert 求最小 6 阶特征值；
3. 粗网格 80×25 重复一遍，Richardson 外推；
4. 打印特征值表，保存场图、特征值图、npz 与文本摘要。

命令行参数（均有默认值，可直接裸跑）：

```bash
python fem_solver.py --phi weights/phi_init.pt --n-eig 6 \
    --fig-dir fem_figures --out-dir fem_results
# --no-coarse    跳过粗网格与 Richardson 外推（只算细网格）
```

自定义配置（Python API）：

```python
from fem_solver import FEMConfig, main

cfg = FEMConfig(phi_path="weights/phi_init.pt", n_eig=10)
res = main(cfg)          # res.fine.omega2[0] 即 ω²_FEM；res.omega2_richardson
```

SDF 不一定是网络权重：`solve_eigen()` 接受任意 `phi_fn(xy)->phi` 可调用对象
（xy 为 (N,2) numpy 物理坐标），例如演化阶段可包装当前 SDF 网络后直接传入：

```python
from fem_solver import FEMConfig, solve_eigen
cfg = FEMConfig()
res = solve_eigen(cfg, phi_fn=my_phi_fn, nx=160, ny=50, label="当前几何")
omega2_fem = res.omega2[0]
```

## 4. 输出解读

### 4.1 打印日志 / fem_summary.txt

- 每个网格一行：节点/自由度数、固支自由度数（160×50 时应为 204 = 2×2×51）、
  S 的范围、∫S dΩ（全材料初始设计 ≈ 0.772，因零水平集压在边界上、
  边界单元半材料，小于域面积 0.8 属正常）、ω²_1 与用时；
- 末尾特征值表：细网格 / 粗网格 / Richardson 三列，**细网格最小值即 ω²_FEM 锚点**。

### 4.2 图片（fem_figures/）

- `fem_fields_mode1.png`：六联图——**u_x、u_y、σ_xx、σ_xy、σ_yy 五个场**
  （均乘 H(SDF) 过滤空洞，有符号场用对称色标 RdBu_r）+ 材料场 S（几何参考）。
  每图叠加 φ=0 零等值线（有内部空洞时）与重块红色虚线框；
  标题给出 ω²_1 的细/粗/Richardson 三个值。
  物理自检：基频模态应为弯曲型——u_y 关于跨中对称、重块处响应最大，
  σ_xx 在固支端上下缘最大，σ_xy 关于跨中反对称。
- `fem_eigenvalues.png`：前 `n_eig` 阶特征值 stem 图（细网格），
  叠加粗网格折线与 Richardson 外推的 ω₁²。

### 4.3 fem_results/fem_result.npz（供 PINN 锚定 / 事后对拍加载）

| 键 | 内容 |
|---|---|
| `omega2_fine` / `omega2_coarse` / `omega2_richardson` | 前 n_eig 阶特征值（三个口径） |
| `modes` | 参考模态，(16422, n_eig) 完整自由度向量，固支处为 0，已质量归一化 |
| `mode1_ux` / `mode1_uy` | 第 1 阶模态位移的节点网格形式 (51, 161) |
| `node_xy` / `elem` / `centers` / `nx` / `ny` / `hx` / `hy` | 细网格网格数据 |
| `free_dofs` / `fixed_dofs` | 自由度划分（固支消去） |
| `elem_S` / `elem_rho` / `elem_E` / `volume_S` | 单元材料场与 ∫S dΩ |

加载示例（与 PINN 模态做 MAC / 加权 L2 对拍）：

```python
import numpy as np
d = np.load("start/fem_results/fem_result.npz")
omega2_fem = float(d["omega2_fine"][0])
ux1, uy1 = d["mode1_ux"], d["mode1_uy"]      # (51, 161) 节点网格
```

## 5. 关键设计说明（实现层面的决策记录）

1. **模板单元加速**：均匀矩形网格下全部单元的 K0/M0 相同，只算一次，
   逐单元乘 E_e/ρ_e。与逐单元 2×2 高斯积分完全等价，只是省去重复计算。
2. **应力在单元中心取值**：σ_e = E_e·C0·B(0,0)·u_e（Q4 的超收敛点），
   不做节点平均；画图时按单元格填充并乘单元中心 S 掩码。
3. **模态符号约定**：特征向量符号任意，统一翻转到最大绝对值分量为正，
   保证多次运行结果可复现、可与 PINN 模态直接对比。
4. **重块用硬指示函数**：单元中心落入 [0.76,0.84]×[0.21,0.29] 即 ρ̂=100，
   与 `mech_init.py` 的 `block_indicator()` 一致。细网格上恰为 8×8 个单元
   （面积 0.0064 精确）；粗网格为 4×5 个单元（面积 0.008，略大），
   是 Richardson 外推误差的来源之一，属预期。
5. **粗/细网格的 S 采样差异**：单元中心采样使粗网格边界环的 S≈0.73、
   细网格≈0.62（零水平集压在边界上的半材料效应），两套网格解的是
   略有差别的有效几何，因此 ω² 不一定按常规方向单调收敛——
   全材料均匀自检（S≡1、无重块）已确认实现本身从上单调收敛
   （0.7119 → 0.7072 → 0.7059 → 0.7055，40×12→320×100）。
6. **空洞区的谱安全性**：ersatz 约定下 E_e/ρ_e ≈ E_solid/ρ̂（S ≫ 1e-6 时），
   深空洞区 S→0 使该比值增大而非减小，不会产生低于物理基频的伪模态。
7. **深空洞数值鲁棒性**：若演化后期 φ < −0.4 使 S 在 float64 下下溢为精确 0，
   对应自由度质量阵出现零行（特征值 +∞，shift-invert 下映射到 OP 谱的 0，
   不影响最小阶求解）；当前初始化阶段 S ≥ 0.5，无此问题。

## 6. 实测基准值（2026-09-03 回填，全材料 + 重块初始设计）

| 量 | 数值 |
|---|---|
| ω²_FEM（细网格 160×50） | **0.22517705** |
| ω²（粗网格 80×25） | 0.19397494 |
| Richardson 外推 | 0.23557776 |
| 前 6 阶（细网格） | 0.225177 / 1.109948 / 2.970576 / 4.764493 / 10.292564 / 10.823200 |

已同步回填：《FEM计算说明.md》§7、《实验设置与计算范围.md》§13、
`mech_init.py` 的 `MechInitConfig.omega2_fem`（原占位 0.238 → 0.22517705）。
注意：`weights/` 下既有的 5 个力学网络权重是按旧锚点 0.238 训练的，
以新锚点为准需重跑 `mech_init.py`。

## 7. 后续衔接

- **PINN 锚定**：`mech_init.py` 的 `omega2_fem` 即本求解器细网格输出；
  交替演化阶段每轮 SDF 更新后重跑 `solve_eigen()` 刷新锚点。
- **模态对拍**：`fem_result.npz` 的 `modes` / `mode1_u*` 供 MAC、加权 L2
  等指标与 PINN 模态对比（`mech_init.py` 诊断中预留了该接口）。
- **SDF 演化阶段**：把当前 SDF 网络包成 `phi_fn`（注意 t 取当前伪时间切片）
  传入 `solve_eigen()` 即可，无需改动本文件。
