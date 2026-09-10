# SDF 初始化代码使用文档

> 本文档对应 `start/sdf_init.py` 的 PyTorch 实现，规格的唯一权威说明见工作区根目录
> 《SDF初始化说明.md》，数值取自《实验设置与计算范围.md》。修改实现时文档需同步修订。

---

## 1. 文件清单

```
start/
├── networks.py          # 网络创建/保存/加载（上一步实现，本步骤复用）
├── sdf_init.py          # 本步骤：SDF 初始化训练（有监督回归，两阶段）
└── （运行后生成）
    ├── sdf_figures/     # 训练过程图片 + 损失曲线 + 最终结果图
    └── weights/
        └── phi_init.pt  # SDF 网络权重（文件名与《网络初始化说明》约定一致）
```

依赖：Python 3 + PyTorch（已在 torch 2.4.1+cpu 上核对）+ matplotlib（仅用于存图，
Agg 后端，无界面要求）。

## 2. 实现内容（与《SDF初始化说明.md》的对应关系）

| 说明文档规格 | 实现 |
|---|---|
| 目标：t=0 切片拟合 φ₀=min(x, Lx−x, y, Ly−y)（有监督回归） | `phi0_target()` + `eval_phi()`（t 恒取 0） |
| offset = 0（零水平集压边界，边界 S=0.5） | `SDFInitConfig.offset`，目标值整体加 offset |
| 损失 = 内部 MSE + w_bnd·边界带 MSE，w_bnd=10 | `sdf_loss()`，`cfg.w_bnd` |
| 边界带采样：按边长比例分配、隔点严格压在边上、外加 4 角点 | `sample_boundary_band()`（带宽取敏感带 4β=0.04） |
| 阶段 A：Adam 每步重采样，lr=2e-3，5000 步，2048+1024 点 | `train_adam()` |
| 阶段 B：L-BFGS 固定点集 16384 Sobol + 4096 边界带，20 块×70 次，强 Wolfe | `train_lbfgs()`（每块新建 `torch.optim.LBFGS`） |
| 独立验证集只评估不训练、早停、回滚到验证最优权重 | `train_lbfgs()` 内（验证集 8192+2048，不同种子） |
| 张量形状逐点对应（目标 (N,1)，不额外 unsqueeze） | `phi0_target` 直接返回 (N,1)，与预测形状一致 |
| float64 | `SDFInitConfig.dtype = torch.float64`，贯穿采样/训练/诊断 |
| 诊断：全域/带内 max 误差、\|∇φ\| 统计、边界 S 值 | `diagnose()` |
| 保存 phi_init.pt | `main()` 末尾 `torch.save(phi.state_dict(), ...)` |

网络结构复用 `networks.py` 的 `build_sdf_network()`（3×64 tanh，输入 (x,y,t)，
xy 通道内置归一化，t 通道不归一化，float64）。

## 3. 快速开始

```bash
cd start
python sdf_init.py
```

全程 CPU 可跑。运行过程：

1. **阶段 A（Adam）**：每步重采样内部点+边界带点。每 **100 步**打印一次损失
   （total / interior / band），每 **500 步**保存一张四联图到 `sdf_figures/`
   （预测 φ、目标 φ₀、绝对误差、材料场 S）。
2. **阶段 B（L-BFGS）**：固定大点集全批量训练，每块最多 70 次内部迭代。
   L-BFGS 没有"每步"的自然粒度，因此**按块**打印（训练损失 + 验证损失 +
   相对改善）并**按块**存图（共 ≤20 块）。连续 3 块验证损失相对改善 < 1e-4
   则早停，结束后自动**回滚到验证损失最优的权重**。
3. **收尾**：保存 `loss_history.png`（损失曲线）、`final.png`（回滚后最终状态）、
   `weights/phi_init.pt`，然后打印诊断结果。

## 4. 输出解读

### 4.1 打印日志

- Adam 段：`total` 应持续下降；若 `total` 卡在 ~Var(φ₀)≈0.02 附近不动，
  说明触发了说明文档 §6.1 的广播坑（本实现已规避，正常不应出现）。
- L-BFGS 段：关注 `val`（独立验证集损失）。`*best` 标记表示当前块刷新了
  最优验证损失；早停后回滚到该最优权重。

### 4.2 诊断（对应说明文档 §7，训练末尾自动打印）

- **拟合误差**：全域 max 与边界带内 max。带内误差应明显小于过渡带宽度
  2β = 0.02，否则材料场 S 在边界附近失真。
- **|∇φ| 统计**：距离函数应处处 ≈ 1；中轴线折点附近被网络抹平、偏小属正常。
- **边界上的 S**：offset=0 时应 ≈ 0.5（半材料）。

### 4.3 图片

`sdf_figures/` 下：`step000000_initial.png`（未训练初始状态）、
`adam_step*.png`（每 500 步）、`lbfgs_block*.png`（每块）、
`loss_history.png`（损失曲线）、`final.png`（最终结果）。

## 5. 在自己的代码中复用

```python
import torch
from sdf_init import SDFInitConfig, phi0_target, sample_boundary_band, eval_phi, main

# 方式一：直接跑完整流程
main()

# 方式二：自定义配置（如缩短 Adam 步数做冒烟测试）
cfg = SDFInitConfig(adam_steps=200, lbfgs_blocks=2,
                    fig_dir="sdf_figures_test", weights_dir="weights_test")
main(cfg)

# 方式三：只取工具函数
cfg = SDFInitConfig()
xy_b = sample_boundary_band(1024, cfg, torch.Generator().manual_seed(0))  # (1028, 2)
tgt = phi0_target(xy_b, cfg)                                              # (1028, 1)
```

加载训练好的 SDF 网络（与 `networks.py` 的约定一致）：

```python
import torch
from networks import NetworkConfig, build_sdf_network

phi = build_sdf_network(NetworkConfig())
phi.load_state_dict(torch.load("weights/phi_init.pt", weights_only=True))
```

## 6. 关键设计说明

1. **边界带定义**：带宽取敏感带宽度 4β = 0.04（《实验设置与计算范围.md》§7），
   带内到边距离 d 的采样中**一半严格为 0**（压在边上，φ₀=0 处误差最要命），
   另一半在 (0, 0.04] 内均匀。四条边按边长比例分配点数（上下边各 1.6/4.2，
   左右边各 0.5/4.2），最大余数法保证总数恰为 n，另加 4 个角点。
2. **Adam 每步重采样**是正则化手段：5000 步见过的点远多于任何固定数据集。
3. **L-BFGS 分块**：每块新建优化器（`max_iter=70`，强 Wolfe 线搜索，
   `history_size=50`），块间检查独立验证集损失；验证集用不同种子的
   Sobol/随机点，只评估不训练，防止过拟合固定配点。
4. **早停与回滚**：连续 `early_stop_patience=3` 块验证损失相对改善
   < `early_stop_rtol=1e-4` 则停止；无论是否早停，最终都回滚到验证损失
   最优的权重再保存。
5. **t 通道**：初始化阶段恒取 0（`eval_phi` 内部拼接零列），演化阶段再启用。

## 7. 后续衔接

训练得到的 `weights/phi_init.pt` 供《应力位移初始化说明》阶段的力学初始化
加载（材料场 S = ½(1+tanh(φ/2β))，β=0.01）。如需把 6 个网络一起保存/加载，
仍使用 `networks.py` 的 `save_networks` / `load_networks`。
