# -*- coding: utf-8 -*-
"""diag_corner_zoom.py — 放大还原左下角零水平集在最初 30 步的侵蚀过程。

回答：角部是被相邻边中段的侵蚀"拖进去"的（侵蚀前沿从侧面向角点推进），
还是角点本身先塌（inflow 边界失控）？
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from networks import NetworkConfig, build_sdf_network
import hjb_step_eik as hjb_mod

DT = 0.005
W = os.path.join(SCRIPT_DIR, "weights")
cfg = hjb_mod.HJBConfig(t_current=0.0, fem_check=False)
dtype = cfg.dtype

phi = build_sdf_network(NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=dtype))

# 角部放大网格 [0,0.2]x[0,0.2]
n = 300
xs = torch.linspace(0.0, 0.2, n, dtype=dtype)
ys = torch.linspace(0.0, 0.2, n, dtype=dtype)
X, Y = torch.meshgrid(xs, ys, indexing="xy")
grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
Xn, Yn = X.numpy(), Y.numpy()

ks = [0, 2, 5, 8, 12, 16, 20, 25, 30]
fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
cmap = plt.get_cmap("viridis")
for ax, k in zip(axes.flat, ks):
    f = os.path.join(W, "phi_init.pt" if k == 0 else f"phi_iter{k}.pt")
    phi.load_state_dict(torch.load(f, weights_only=True))
    phi.eval()
    with torch.no_grad():
        tt = torch.full((grid.shape[0], 1), k * DT, dtype=dtype)
        z = phi(torch.cat([grid, tt], dim=1)).reshape(X.shape).numpy()
    ax.pcolormesh(Xn, Yn, z, cmap="RdYlGn", shading="auto", vmin=-0.05, vmax=0.05)
    ax.contour(Xn, Yn, z, levels=[0.0], colors=["k"], linewidths=1.6)
    ax.set_title(f"k={k} (t={k*DT:.3f})", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(0, 0.2); ax.set_ylim(0, 0.2)
fig.suptitle("左下角零水平集侵蚀过程（黑线=φ=0，绿=材料，红=空洞）")
out = os.path.join(SCRIPT_DIR, "diag_figures", "corner_zoom.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120)
print(f"已保存 {out}")

# 定量：底边 (y=0) 与左边 (x=0) 上零水平集位置随 k 的变化（角帽宽度的收缩）
print(f"\n{'k':>3} | 底边上 phi 变号位置 s*（角帽右端） | 左边上 phi 变号位置 s*（角帽上端）")
s = torch.linspace(0.0, 0.2, 2001, dtype=dtype).unsqueeze(1)
for k in ks:
    f = os.path.join(W, "phi_init.pt" if k == 0 else f"phi_iter{k}.pt")
    phi.load_state_dict(torch.load(f, weights_only=True))
    phi.eval()
    res = []
    for pts in [torch.cat([s, torch.zeros_like(s)], 1), torch.cat([torch.zeros_like(s), s], 1)]:
        with torch.no_grad():
            tt = torch.full((pts.shape[0], 1), k * DT, dtype=dtype)
            pv = phi(torch.cat([pts, tt], dim=1)).squeeze(1).numpy()
        # 找 phi 从负（角点侧）转正（域内侧）的位置 = 角帽材料起点
        idx = np.where(np.diff(np.sign(pv)) > 0)[0]
        res.append(f"{float(s[idx[0]]):.4f}" if len(idx) else "全负(角帽消失)")
    print(f"{k:>3} | {res[0]:>22} | {res[1]:>22}")
