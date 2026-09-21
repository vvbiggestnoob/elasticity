# -*- coding: utf-8 -*-
"""diag_erosion_timeline.py — 还原 92 步运行中角部/边中段的侵蚀时间线与应变能演变。

回答：角部何时开始被侵蚀？彼时角部应变能是多少？V_n 在角部何时转负？
左右边中段何时开始侵蚀？
"""
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from networks import NetworkConfig, build_mechanics_networks, build_sdf_network, MECH_NET_NAMES
import hjb_step_eik as hjb_mod

DT = 0.005   # 该次运行的 dt（loop_history.json 里 n_iter=1000, dt=0.005）
W = os.path.join(SCRIPT_DIR, "weights")

cfg = hjb_mod.HJBConfig(t_current=0.0, fem_check=False)
dtype = cfg.dtype

probes = {
    "角(0,0)": (0.0, 0.0),
    "近角(.02,.02)": (0.02, 0.02),
    "左边中(0,.25)": (0.0, 0.25),
    "下边中(0.4,0)": (0.4, 0.0),
    "重块心(.8,.25)": (0.8, 0.25),
}
xy = torch.tensor([p for p in probes.values()], dtype=dtype)

phi = build_sdf_network(NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=dtype))
net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, hidden_layers=cfg.mech_hidden_layers,
                        hidden_width=cfg.mech_hidden_width, dtype=dtype)
mech = build_mechanics_networks(net_cfg)

print(f"{'k':>4} | " + " | ".join(f"{n:>14}" for n in probes) + " | 角eng | 左中eng")
ks = list(range(0, 92, 5)) + [91]
for k in ks:
    t_k = k * DT
    phi_file = os.path.join(W, "phi_init.pt" if k == 0 else f"phi_iter{k}.pt")
    mech_suf = "_init.pt" if k == 0 else f"_iter{k}.pt"
    if not os.path.exists(phi_file):
        continue
    phi.load_state_dict(torch.load(phi_file, weights_only=True))
    phi.eval()
    with torch.no_grad():
        tt = torch.full((xy.shape[0], 1), t_k, dtype=dtype)
        pv = phi(torch.cat([xy, tt], dim=1)).squeeze(1)
    # 力学：角部/左中应变能
    eng_str = ""
    mech_ok = all(os.path.exists(os.path.join(W, n + mech_suf)) for n in MECH_NET_NAMES)
    if mech_ok:
        for n in MECH_NET_NAMES:
            mech[n].load_state_dict(torch.load(os.path.join(W, n + mech_suf),
                                               weights_only=True))
        mech.eval()
        mech.requires_grad_(False)
        # 角部取近角点小邻域峰值（与 corner_strain_metric 同法，单边 0.05）
        s = torch.linspace(0.0, 0.05, 32, dtype=dtype).unsqueeze(1)
        z = torch.zeros_like(s)
        pts_corner = torch.cat([torch.cat([s, z], 1), torch.cat([z, s], 1)], 0)
        pts_all = torch.cat([pts_corner, xy[2:3]], 0)   # 角部探针 + 左边中
        with torch.no_grad():
            phi_val = hjb_mod.eval_phi_at(phi, pts_all, t_k)
            _, _, _, e = hjb_mod.fields_from_phi_val(phi_val, pts_all, cfg)
        _, _, ex, ey, gxy = hjb_mod.frozen_strain(mech, pts_all)
        sxx, syy, sxy = hjb_mod.constitutive_stress(ex, ey, gxy, e, cfg)
        eng = (sxx * ex + syy * ey + sxy * gxy).squeeze(1)
        eng_str = f"{float(eng[:64].max()):.3f} | {float(eng[64]):.3f}"
    print(f"{k:>4} | " + " | ".join(f"{float(v):>14.4f}" for v in pv)
          + f" | {eng_str}")
