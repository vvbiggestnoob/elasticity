# HJB 网格求解（对照基准）代码使用文档

> 本文档对应 `start/hjb_fem.py`，是 PINN 版 HJB 演化（`start/hjb_step.py`）的
> **网格对照基准**。规格的唯一权威说明见《HJB方程优化说明.md》。
> 修改实现时文档需同步修订。

---

## 1. 定位与文件清单

`hjb_fem.py` 与 `hjb_step.py` 解**同一个初值问题**——同一冻结力学网络、同一
SDF 初始条件、同一 V_n 场（全部复用 `hjb_step.py` 的实现，逐点一致）——唯一
差别是 HJB 方程的求解器：

| | `hjb_step.py`（PINN 版） | `hjb_fem.py`（本版，对照基准） |
|---|---|---|
| 解的表示 | SDF 网络 φ(x,y,t)，时空整体拟合 | 160×50 规则网格上的 φ 值，时间推进 |
| 求解方式 | 软损失训练（L_r + L_0 + L_anchor） | 一阶 Godunov 迎风 + 显式 Euler |
| 边界 | 无显式边界（出流式） | 无显式边界（内部单侧差分） |
| 用途 | 正式演化（输出光滑 SDF 网络） | 交叉验证 PINN 解的正确性 |

> 名称中的 "FEM" 是沿用习惯叫法；HJB 是双曲方程，实际离散为**有限差分**
> （Godunov 迎风），不是有限元。网格与 `fem_solver.py` 的细网格一致（160×50）。

```
start/
├── hjb_step.py          # PINN 版（被复用：HJBConfig、V_n、加载、积分）
├── hjb_fem.py           # 本步骤：网格对照基准
└── （运行后生成）
    └── hjb_fem_results/
        ├── hjb_fem_sdf.npz      # 网格坐标、φ(0)、V_n、演化后 φ(t_new)
        ├── hjb_fem_state.json   # 标量记录（ω²、面积前后、子步数、对比指标）
        ├── hjb_fem_evolution.png    # 演化前后 φ/S/ΔS + 零水平集对比
        └── hjb_fem_vs_pinn.png      # 若存在 PINN 结果则自动对照
```

依赖：PyTorch（复用 hjb_step 的网络前向）+ numpy（网格推进）+ matplotlib。
运行前需已有 `weights/phi_init.pt` 与 5 个 `weights/*_init.pt`。

## 2. 数值格式

**方程**（与 PINN 版同符号约定）：∂φ/∂t = V_n·|∇φ|，初值 φ(0) = phi_ref(t=0 切片)。

**空间离散**：一阶 Godunov 迎风。写 φ_t + F|∇φ| = 0（F = −V_n），逐点按符号：

- V_n < 0（F > 0）：|∇φ|² = max(D⁻ˣ,0)² + min(D⁺ˣ,0)² + max(D⁻ʸ,0)² + min(D⁺ʸ,0)²
- V_n > 0（F < 0）：|∇φ|² = min(D⁻ˣ,0)² + max(D⁺ˣ,0)² + min(D⁻ʸ,0)² + max(D⁺ʸ,0)²

**边界**：无显式边界条件（出流式）——边界节点缺失的差分用内部单侧差分替代，
边界值由 PDE + 内部信息决定，零水平集可自由后退/外扩。
（2026-09-08 起不再使用零法向梯度 ghost cell，与 PINN 版移除 L_b 一致。）

**时间推进**：显式 Euler，子步由 CFL 条件自适应：

$$
n_{sub} = \Bigl\lceil \frac{\max|V_n|\cdot dt}{cfl\cdot\min(h_x,h_y)} \Bigr\rceil,
\qquad cfl = 0.9,\quad h_x = h_y = 0.01
$$

（全材料阶段 max|V_n| ≈ 25（重块处），dt = 0.005 时约 15 个子步。）

## 3. 快速开始

```bash
cd start
python hjb_fem.py
```

秒级完成。流程：加载力学网络与 SDF（与 PINN 版同一实现）→ 算 ω²、面积 →
在网格上评估 φ(0) 与 V_n → Godunov 推进 → 保存网格 SDF 与图 → 若存在
`weights/phi_iter1.pt`（PINN 结果）自动加载并对照。

**时间区间同步**：对照必须在同一时间区间 [t_current, t_current+dt] 上进行。
若存在 PINN 演化记录（`weights/hjb_state.json`），脚本自动以其 `t_current`/`dt`
为准（打印"[同步]..."提示），覆盖配置默认值——防止 `hjb_step.py` 的 dt 被改过
但 phi_iter1.pt 尚未重跑时两版时间区间不一致。

自定义配置：

```python
from hjb_fem import HJBFEMConfig, main

cfg = HJBFEMConfig(nx=160, ny=50, cfl=0.9,
                   phi_pinn_path="weights/phi_iter1.pt",
                   out_dir="hjb_fem_results_test")
main(cfg)
```

## 4. 输出解读

### 4.1 打印日志

- **V_n（网格）统计**：应与 PINN 版打印的池点统计一致（同一实现）；
- **子步数**：由 max|V_n| 与 CFL 决定；
- **面积（网格口径）**：节点均值 × 域面积。注意网格口径与 PINN 的 MC 口径
  略有差异（β=0.01 的过渡带在 160×50 网格上只有 2 个单元）；
- **对照指标**（存在 PINN 结果时）：|Δφ| max/mean、敏感带（|φ₀|≤4β）|Δφ| max、
  |ΔS| max/mean、两版面积差。

### 4.2 文件

- `hjb_fem_sdf.npz`：`xs, ys`（节点坐标）、`phi0`、`vn`、`phi_fem`（均为
  (ny+1, nx+1) 网格数组）及标量（t_current, dt, t_new, omega2, n_sub, dt_sub）；
- `hjb_fem_state.json`：标量记录 + 对照指标（`pinn_compare` 字段）；
- `hjb_fem_evolution.png`：与 PINN 版 `final.png` 同版式，可直接对比；
- `hjb_fem_vs_pinn.png`：φ_FEM vs φ_PINN（含各自零水平集）、Δφ、ΔS 分布。

## 5. 关键设计说明

1. **对照的有效性**：V_n、ω²、面积、初始 φ 全部复用 `hjb_step.py` 的实现
   （`compute_vn`、`rayleigh_and_area`、`eval_phi_at` 等），两版唯一差别是
   HJB 求解器——观察到的差异即求解器误差（PINN 的软约束/网络光滑性 vs
   网格的离散/数值扩散）。
2. **网格版自身误差**：一阶格式有 O(hx) 数值扩散；β=0.01 的过渡带只有
   2 个单元，S 的分辨较粗；它是"同一初值问题的忠实参考"，但不是无误差真解。
3. **制造解锚点的关系**：PINN 版的 L_anchor（一阶 Taylor 制造解）与本网格解
   在一阶意义下等价——开启锚点后，PINN 解应明显向网格解靠拢。
4. **历史**：2026-09-08 前边界为零法向梯度 ghost cell（边界值被钉死，
   零水平集无法从域边界后退），已随 L_b 移除改为出流式单侧差分。

## 6. 后续衔接

- 每次修改 PINN 版（`hjb_step.py`）的损失或训练配置后，重跑两版即可用
  本脚本定量检验 PINN 解是否仍与网格参考一致；
- 外层交替迭代阶段，本脚本同样适用于每一轮 HJB 步的对照（改 `phi_path` /
  `phi_pinn_path` 指向当轮文件即可）。
