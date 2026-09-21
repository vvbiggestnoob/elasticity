# PINN-HJB 特征值拓扑优化

用神经网络（PINN）+ HJB 方程交替迭代的方法，求解弹性结构**最大化最低固有频率**（最大化最小特征值 ω²）的拓扑优化问题，体积约束 50%。

- **物理问题**：设计域 Ω = [0, 1.6] × [0, 0.5]，左/右端固支，上/下自由，中央有不可优化的固定重块（[0.76, 0.84]×[0.21, 0.29]，密度 100）
- **方法**：6 个全连接网络（5 个力学：u_x, u_y, σ_xx, σ_xy, σ_yy + 1 个 SDF 网络 φ），SDF 经 Heaviside 插值给出材料场；HJB 方程驱动界面演化，FEM 提供特征值锚点对拍
- **详细方法与损失函数**：见 [docs/方法整理_训练流程与损失函数.md](docs/方法整理_训练流程与损失函数.md)（以代码为准的权威文档）

## 目录结构

```
├── docs/                     # 说明文档（理论推导、方案设计、实验设置）
│   ├── problem.md            # 问题定义
│   ├── 方案_详细版.md         # 设计方案
│   ├── 方法整理_训练流程与损失函数.md  # ★ 整体方法与训练流程（以代码为准）
│   ├── 实验设置与计算范围.md
│   ├── FEM计算说明.md / HJB方程优化说明.md / HJB方程推导与特征值拓扑优化.md
│   ├── SDF初始化说明.md / 应力位移初始化说明.md / 网络初始化说明.md / 重初始化方案说明.md
│   ├── DNN_Level_Set_Topology_Optimization_中文翻译.md  # 参考论文翻译
│   └── code_usage/           # 各模块代码使用文档与损失函数说明
├── references/               # 参考论文 PDF
├── start/                    # 核心代码
│   ├── networks.py           # 6 个网络定义（内置归一化，float64）
│   ├── sdf_init.py           # 阶段 0：SDF 初始化
│   ├── mech_init.py          # 阶段 0 + Step A：力学网络训练（7 项损失）
│   ├── hjb_step_eik.py       # Step B：HJB 单步演化（含 eikonal 正则与角部冻结）
│   ├── alternating_loop.py   # 主循环：变量传递、罚参数更新、汇总
│   ├── fem_solver.py         # FEM 特征值对拍（锚点来源）
│   ├── hjb_fem.py            # HJB 网格法求解（对照）
│   ├── diag_*.py             # 诊断/消融脚本
│   ├── weights/              # 初始化结果：*_init.pt + 状态/历史 json
│   ├── weights_v2/           # v2 力学初始化结果
│   ├── fem_results/          # FEM 对拍结果（npz + 摘要）
│   └── hjb_fem_results/      # HJB 网格法结果（npz + 状态）
└── exp01_bridge/             # 当前实验：角部硬非设计域桥式结构
    ├── run_bridge.py         # 实验配置入口（corner_freeze=0.08, λ₀=10, dt=0.015）
    ├── check_progress.py     # 进度检查
    ├── run_bridge.log        # 运行日志
    └── weights/              # 本实验的初始化结果与循环历史
```

## 运行流程

```bash
# 阶段 0：一次性初始化（start/ 目录）
python start/sdf_init.py      # SDF 网络拟合全材料初始形状 → weights/phi_init.pt
python start/fem_solver.py    # FEM 特征值对拍，给出锚点 ω²_FEM
python start/mech_init.py     # 初始几何上训练 5 个力学网络 → weights/*_init.pt

# 主循环（当前实验配置）
python exp01_bridge/run_bridge.py [n_iter] [corner_freeze] [dt]
```

主循环每次迭代：HJB 演化（冻结力学，φ 热启动）→ FEM 对拍更新锚点 → 罚参数自适应更新 → 力学网络在新几何上热启动重解。

## 说明

- 本仓库**不包含训练过程图片**（`loop_figures/`、`*_figures/` 等）与**逐迭代权重快照**（`*_iter*.pt`），仅保留初始化结果与关键状态数据；完整产物在本地实验目录生成。
- 依赖：PyTorch（全程 float64）、NumPy、SciPy（FEM 稀疏特征值）、Matplotlib。
