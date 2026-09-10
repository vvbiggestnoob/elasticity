# -*- coding: utf-8 -*-
"""
最小使用示例：创建 6 个网络、前向传播、对物理坐标求一阶导、保存权重。

运行：python example_usage.py
"""

import torch

from networks import NetworkConfig, build_all_networks, save_networks

# 1. 创建全部 6 个网络（5 力学 + 1 SDF），默认 lx=1.6, ly=0.5，float64
cfg = NetworkConfig()
bundle = build_all_networks(cfg)
mech, phi = bundle.mechanics, bundle.phi

# 2. 前向传播：对外接口始终是物理坐标 (x, y)，归一化在网络内部完成
xy = torch.tensor([[0.4, 0.1], [0.8, 0.25], [1.2, 0.4]], dtype=torch.float64)
print("u_x(x, y) =\n", mech["u_x"](xy))
print("sigma_xx(x, y) =\n", mech["sigma_xx"](xy))

# SDF 网络多一个伪时间维 t（初始化阶段恒取 0），输入为 (x, y, t)
xyt = torch.cat([xy, torch.zeros(len(xy), 1, dtype=torch.float64)], dim=1)
print("phi(x, y, t=0) =\n", phi(xyt))

# 3. 一阶自动微分：链式法则穿过内置固定仿射，得到的是对物理坐标的导数
xy_grad = xy.clone().requires_grad_(True)
u = mech["u_x"](xy_grad)
(du,) = torch.autograd.grad(u.sum(), xy_grad)
print("du_x/dx, du_x/dy =\n", du)  # 第 0 列 d/dx，第 1 列 d/dy

# 4. 保存权重（每个网络一个文件：u_x_init.pt ... phi_init.pt）
paths = save_networks(bundle, out_dir="weights")
print("已保存：")
for p in paths:
    print("  ", p)
