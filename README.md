# PINN + HJB 弹性特征值拓扑优化（可移植包）

基于物理信息神经网络（PINN）与 HJB 方程的弹性特征值拓扑优化项目。纯 Python / PyTorch 实现，无外部 FEM 软件依赖。

## 包结构

```
paper_elastic_portable/
├── README.md            ← 本文件
├── requirements.txt     ← Python 依赖
├── start/               ← 实验代码（主目录）
│   ├── alternating_loop.py   ← 主入口：交替循环（Step A 力学重解 ↔ Step B HJB 演化）
│   ├── networks.py           ← 网络基础库（6 个 MLP，被所有模块 import）
│   ├── sdf_init.py           ← 阶段1：SDF 初始化训练
│   ├── mech_init.py          ← 阶段2：力学网络初始化
│   ├── mech_init_ipm.py      ← 力学网络初始化（反幂法版本）
│   ├── hjb_step_eik.py       ← HJB 演化（含 eikonal 正则，当前默认版本）
│   ├── hjb_step.py           ← HJB 演化（无 eikonal 旧版，备用）
│   ├── hjb_fem.py            ← HJB 网格数值解（Godunov 迎风，PINN 对照基准）
│   ├── fem_solver.py         ← FEM 特征值基准求解器（Q4 单元）
│   ├── compare_energy_balance.py / plot_energy_balance.py  ← 分析绘图脚本
│   ├── diag_edge_vn.py       ← 诊断脚本
│   ├── verify_networks.py / example_usage.py / _smoke_loop.py ← 自检/示例/冒烟测试
│   └── weights/              ← 初始化权重（已训练好，见下）
├── start_files/         ← 各代码模块的使用文档（10 篇，最重要的见下）
├── docs/                ← 说明文档（问题定义、技术方案、理论推导、各初始化说明等 11 篇）
└── references/          ← 参考论文 PDF 与推导手稿图
```

## 新电脑运行（无需初始化）

本包已内置 6 个初始化权重（`start/weights/` 下的 `phi_init.pt`、`u_x_init.pt`、`u_y_init.pt`、`sigma_xx_init.pt`、`sigma_xy_init.pt`、`sigma_yy_init.pt`），主循环启动时自动加载，**跳过全部初始化阶段**。

```bash
pip install -r requirements.txt   # torch numpy scipy matplotlib，Python 3.8+
cd start
python alternating_loop.py        # 直接跑主循环
```

可选验证：

```bash
python verify_networks.py         # 网络自检（秒级）
python _smoke_loop.py             # 冒烟测试（小规模跑通整个流程）
python fem_solver.py              # 重算 FEM 锚点（ω²_FEM ≈ 0.22517705，已内置为默认值）
```

## 如需从头重新初始化（一般不需要）

仅当删除了 `start/weights/` 下的 init 权重、或修改了几何参数（lx/ly）时才需要，按顺序执行：

```bash
cd start
python sdf_init.py        # 生成 weights/phi_init.pt
python fem_solver.py      # 重算 ω²_FEM 锚点，并更新 mech_init.py 中的 omega2_fem 默认值
python mech_init.py       # 生成 5 个 *_init.pt
python alternating_loop.py
```

## 说明

- 所有路径均相对于脚本所在目录自动解析，输出目录（`weights/`、`loop_figures/` 等）由代码自动创建，整个文件夹可随意搬移。
- 数值参数的权威来源见 `docs/实验设置与计算范围.md`；运行前提与变量传递约定见 `start_files/交替循环代码使用文档.md`。
- 训练过程图、迭代中间权重等产物未包含在本包内，重新运行即可再生成。
