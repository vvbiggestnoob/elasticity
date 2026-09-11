# PINN-HJB 特征值拓扑优化

## 文件结构

```
part1_networks_and_tools.py   # 网络、FEM、工具函数
part2_initialization.py       # 初始化训练（SDF + 力学网络）
part3_hjb_reinit.py           # HJB 演化 + 重初始化（暂不执行）
part4_main_loop.py            # 主循环（暂不执行）
part5_plotting.py             # 可视化
```

## 运行步骤

### 第一步：初始化训练

```bash
python part2_initialization.py
```

这会：
1. 创建 6 个网络（5 个力学 + 1 个 SDF）
2. 训练 SDF 网络（拟合全材料初始形状）
3. 训练力学网络（PINN 求解特征值问题）
4. 保存网络权重到 `*.pt` 文件
5. 保存训练历史到 `mech_loss_history.npy`

**预计时间**：10-30 分钟（取决于 CPU）

### 第二步：查看结果

```bash
python part5_plotting.py
```

这会加载训练好的网络并生成以下图片到 `outputs/` 目录：

| 图片 | 内容 |
|------|------|
| `sdf.png` | SDF 等值线图 |
| `material.png` | 材料分布图（含重块位置标注） |
| `displacement.png` | 位移场（\|u\|, u_x, u_y） |
| `stress.png` | 应力场（σ_xx, σ_yy, σ_xy, Von Mises） |
| `convergence.png` | 训练收敛历史 |

## 参数说明

### 几何参数（part1_networks_and_tools.py）

```python
LX = 1.6          # 设计域长度（水平方向）
LY = 0.5          # 设计域高度（垂直方向）
BLOCK_X0, BLOCK_X1 = 0.76, 0.84   # 重块 x 范围
BLOCK_Y0, BLOCK_Y1 = 0.21, 0.29   # 重块 y 范围
BLOCK_RHO = 100.0                  # 重块密度
```

### 材料参数

```python
E_SOLID = 1.0     # 固体弹性模量
E_VOID = 1e-6     # 孔洞弹性模量
NU = 0.3          # 泊松比
BETA = 0.01       # Heaviside 过渡带宽度
```

### 训练参数（part2_initialization.py）

```python
# SDF 初始化
n_points=5000     # 每次采样点数
lr=1e-3           # 学习率
max_steps=2000    # 最大迭代步数

# 力学网络训练
n_domain=2000     # 域内采样点数
n_boundary=100    # 边界采样点数
adam_steps=5000   # Adam 迭代步数
lbfgs_steps=500   # L-BFGS 迭代步数
```

## 验证要点

训练完成后，检查以下内容：

1. **SDF 图**：初始 SDF 应该是到边界的距离，域内为正
2. **材料分布图**：全材料（S≈1），重块位置标注正确
3. **位移场**：位移应该在固定端（左右）为零
4. **应力场**：应力应该在自由端（上下）为零
5. **收敛历史**：各损失项应该下降，Rayleigh 商应该接近 FEM 参考值

## 注意事项

- 当前版本**不包含 SDF 演化**（HJB 方程），只做初始化训练
- 孔洞边界通过 ersatz material 自动满足（E_void = 1e-6）
- 材料掩码统一使用 S^1
- SDF 网络输入为 (x, y, t)，初始化时 t=0

## 后续工作

如需进行拓扑优化（SDF 演化），需要：
1. 实现 HJB 演化（part3_hjb_reinit.py）
2. 实现重初始化（part3_hjb_reinit.py）
3. 实现主循环（part4_main_loop.py）
