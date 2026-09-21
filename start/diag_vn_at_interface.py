# -*- coding: utf-8 -*-
"""diag_vn_at_interface.py — 在零水平集上（而非域边界上）测 V_n，回答：
初始几何下角部倒角处 / 左边界中段的 V_n 到底是正是负？V_n 转正/负的位置
与零水平集的相对关系如何？

用法：python diag_vn_at_interface.py [k]   （k=0 用 phi_init + *_init，否则 phi_iter{k} + *_iter{k}）
"""
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from networks import NetworkConfig, build_mechanics_networks, build_sdf_network, MECH_NET_NAMES
import hjb_step_eik as hjb_mod

DT = 0.005
k = int(sys.argv[1]) if len(sys.argv) > 1 else 0
W = os.path.join(SCRIPT_DIR, "weights")

cfg = hjb_mod.HJBConfig(t_current=k * DT, fem_check=False)
dtype = cfg.dtype

phi = build_sdf_network(NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=dtype))
phi.load_state_dict(torch.load(os.path.join(
    W, "phi_init.pt" if k == 0 else f"phi_iter{k}.pt"), weights_only=True))
phi.eval()
phi.requires_grad_(False)

net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, hidden_layers=cfg.mech_hidden_layers,
                        hidden_width=cfg.mech_hidden_width, dtype=dtype)
mech = build_mechanics_networks(net_cfg)
suf = "_init.pt" if k == 0 else f"_iter{k}.pt"
for n in MECH_NET_NAMES:
    mech[n].load_state_dict(torch.load(os.path.join(W, n + suf), weights_only=True))
mech.eval()
mech.requires_grad_(False)

# ω² 与面积（与 HJB 步内一致的大点集）
xy_int = hjb_mod.sample_interior_sobol(65536, cfg, 7)
omega2, area = hjb_mod.rayleigh_and_area(mech, phi, k * DT, xy_int, cfg)
penalty = (area - cfg.v_target) / cfg.alpha
print(f"k={k}  t={k*DT:.3f}  ω²={omega2:.4f}  area={area:.4f}  "
      f"罚压力 p={penalty:.3f}（注：循环内实际 p 由 Step D 自适应给出，此处 α=1）")

# --- 近角二维扫描：V_n 符号 vs 零水平集 -------------------------------
n1 = 400
xs = torch.linspace(0.0, 0.12, n1, dtype=dtype)
ys = torch.linspace(0.0, 0.12, n1, dtype=dtype)
X, Y = torch.meshgrid(xs, ys, indexing="xy")
grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
with torch.no_grad():
    pv = hjb_mod.eval_phi_at(phi, grid, k * DT).squeeze(1)
vn, eng, kin = hjb_mod.compute_vn(mech, phi, grid, cfg, omega2, area)
vn = vn.squeeze(1)
eng = eng.squeeze(1)

# 沿底边 (y=0) 与左边 (x=0) 的 V_n 剖面 + 各点的 phi 符号
def profile(name, pts):
    with torch.no_grad():
        p = hjb_mod.eval_phi_at(phi, pts, k * DT).squeeze(1)
    v, e, _ = hjb_mod.compute_vn(mech, phi, pts, cfg, omega2, area)
    v, e = v.squeeze(1), e.squeeze(1)
    print(f"\n{name}（s | phi | V_n | eng）:")
    for i in range(0, pts.shape[0], max(1, pts.shape[0] // 25)):
        print(f"  s={float(pts[i, 0] + pts[i, 1]):.4f}  phi={float(p[i]):+.4f}  "
              f"V_n={float(v[i]):+.4f}  eng={float(e[i]):.4f}")

s1 = torch.linspace(0.0, 0.15, 151, dtype=dtype).unsqueeze(1)
profile("底边近角 (y=0)", torch.cat([s1, torch.zeros_like(s1)], dim=1))
profile("左边近角 (x=0)", torch.cat([torch.zeros_like(s1), s1], dim=1))

# 零水平集与 V_n=0 等值线在近角区域的位置（沿对角线 y=x）
d = torch.linspace(0.0, 0.12, 1201, dtype=dtype)
diag = torch.stack([d, d], dim=1)
with torch.no_grad():
    pd = hjb_mod.eval_phi_at(phi, diag, k * DT).squeeze(1)
vd, ed, _ = hjb_mod.compute_vn(mech, phi, diag, cfg, omega2, area)
vd = vd.squeeze(1)
# 找 phi=0 与 V_n=0 的对角线位置
def first_cross(val, target=0.0):
    sign = torch.sign(val - target)
    idx = (sign[1:] * sign[:-1] < 0).nonzero()
    return float(d[idx[0]]) if len(idx) else None
z_phi = first_cross(pd)
z_vn = first_cross(vd)
print(f"\n对角线 y=x 上：phi=0 在 d={z_phi}，V_n=0 在 d={z_vn}")
print(f"（d 为距角点 (0,0) 的坐标，实际距离 = d·√2）")
if z_phi is not None and z_vn is not None:
    if z_vn < z_phi:
        print(f"→ V_n 在零水平集内侧 {z_phi - z_vn:.4f} 处转负：倒角处 V_n<0，角部会被侵蚀")
    else:
        print(f"→ V_n 在零水平集外侧 {z_vn - z_phi:.4f} 处才转负：倒角处 V_n>0，角部应受保护")
