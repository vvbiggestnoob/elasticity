# -*- coding: utf-8 -*-
"""diag_fem_vs_pinn_corners.py — 初始几何上 FEM 与 PINN 的边界应变能密度对拍。

回答的问题：PINN 力学网络是否低估了四角等关键位置的应变能密度
（这会导致 V_n 在角部被低估、角部被误侵蚀）。两者均用质量归一化一阶模态。
"""
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fem_solver import FEMConfig, solve_eigen, load_sdf_fn, mode_fields
from networks import NetworkConfig, build_mechanics_networks, build_sdf_network, MECH_NET_NAMES
import hjb_step_eik as hjb_mod

cfg_f = FEMConfig(do_coarse=False)
phi_fn = load_sdf_fn(cfg_f)                      # phi_init.pt（全材料 SDF）
res = solve_eigen(cfg_f, phi_fn, cfg_f.nx_fine, cfg_f.ny_fine, label="初始几何 FEM")

# FEM：单元中心应变能密度 eng = ε·σ（σ 已含 E_e），动能密度 kin = ω²ρ|u_c|²
u = res.modes[:, 0]
u_e = u[res.edof]                                # (E, 8)
eps = u_e @ res.B_center.T                       # (E, 3)
f = mode_fields(res, cfg_f, 0)
eng_fem = (eps * f.sig).sum(axis=1)
ux_c = u_e[:, 0::2].mean(axis=1)
uy_c = u_e[:, 1::2].mean(axis=1)
kin_fem = res.omega2[0] * res.rho * (ux_c ** 2 + uy_c ** 2)
centers = res.mesh.centers

# PINN：同一批单元中心点，用冻结力学网络算 eng / kin
# 用法：python diag_fem_vs_pinn_corners.py [weights_dir] [suffix]
w_dir = sys.argv[1] if len(sys.argv) > 1 else "weights"
w_suf = sys.argv[2] if len(sys.argv) > 2 else "_init.pt"
print(f"PINN 权重：{w_dir}/*{w_suf}")
cfg_p = hjb_mod.HJBConfig(t_current=0.0, fem_check=False)
net_cfg = NetworkConfig(lx=cfg_p.lx, ly=cfg_p.ly,
                        hidden_layers=cfg_p.mech_hidden_layers,
                        hidden_width=cfg_p.mech_hidden_width, dtype=cfg_p.dtype)
mech = build_mechanics_networks(net_cfg)
for n in MECH_NET_NAMES:
    mech[n].load_state_dict(torch.load(os.path.join(
        SCRIPT_DIR, w_dir, n + w_suf), weights_only=True))
mech.eval()
mech.requires_grad_(False)
phi = build_sdf_network(NetworkConfig(lx=cfg_p.lx, ly=cfg_p.ly, dtype=cfg_p.dtype))
phi.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "weights", "phi_init.pt"),
                               weights_only=True))
phi.eval()
phi.requires_grad_(False)

xy = torch.as_tensor(centers, dtype=cfg_p.dtype)
omega2_pinn, area_pinn = hjb_mod.rayleigh_and_area(
    mech, phi, 0.0, hjb_mod.sample_interior_sobol(65536, cfg_p, 7), cfg_p)
_, eng_p, kin_p = hjb_mod.compute_vn(mech, phi, xy, cfg_p, omega2_pinn, area_pinn)
eng_pinn = eng_p.squeeze(1).numpy()
kin_pinn = (omega2_pinn * kin_p).squeeze(1).numpy()

print(f"\nω²：FEM = {res.omega2[0]:.6f} | PINN Rayleigh = {omega2_pinn:.6f}")
print(f"{'位置':<12} {'FEM eng':>10} {'PINN eng':>10} {'比值':>8} "
      f"{'FEM kin':>10} {'PINN kin':>10}")
probes = [("左下角", (0.0, 0.0)), ("右下角", (cfg_f.lx, 0.0)),
          ("左上角", (0.0, cfg_f.ly)), ("右上角", (cfg_f.lx, cfg_f.ly)),
          ("左边中", (0.0, 0.25)), ("右边中", (cfg_f.lx, 0.25)),
          ("左边 y=0.1", (0.0, 0.1)), ("左边 y=0.4", (0.0, 0.4)),
          ("上边中", (0.8, cfg_f.ly)), ("下边中", (0.8, 0.0)),
          ("重块中心", (0.8, 0.25))]
for name, c in probes:
    i = int(np.argmin(((centers - np.array(c)) ** 2).sum(axis=1)))
    ratio = eng_pinn[i] / max(eng_fem[i], 1e-30)
    print(f"{name:<12} {eng_fem[i]:>10.3f} {eng_pinn[i]:>10.3f} {ratio:>8.2f} "
          f"{kin_fem[i]:>10.4f} {kin_pinn[i]:>10.4f}")
