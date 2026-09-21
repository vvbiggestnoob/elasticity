# -*- coding: utf-8 -*-
"""实测：初始条件权重 lam_0 增大能否守住左边界（当前几何 phi_iter1，dt=0.005）。

固定重跑第 2 步（phi_1 → phi_2'），只改 lam_0，报告左边界中段 φ 与总面积变化。
关注：lam_0 增大是否守住左边界缺口，以及是否过度锚定拖住总面积侵蚀。

用法：python diag_lam0_sweep.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

import hjb_step_eik as hjb_mod
from networks import NetworkConfig, build_sdf_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(SCRIPT_DIR, "..", "exp01_bridge", "weights")
TMP = os.path.join(SCRIPT_DIR, "diag_figures", "lam0_sweep_tmp")

BASE = dict(
    t_current=0.005, dt=0.005, alpha=0.562622, v_target=0.4,
    corner_freeze=0.08, w_corner_freeze=10.0,
    phi_path=os.path.join(WEIGHTS, "phi_iter1.pt"),
    mech_dir=WEIGHTS, mech_suffix="_iter1.pt",
    adam_steps=800, lbfgs_blocks=30,
    fem_check=False, quiet_figures=True,
    dtype=torch.float64,
)

LAM0_LIST = [0.1, 1.0, 3.0, 10.0]


def edge_stat(phi_path: str, t_new: float):
    phi = build_sdf_network(NetworkConfig(dtype=torch.float64))
    phi.load_state_dict(torch.load(phi_path, weights_only=True))
    phi.eval().requires_grad_(False)
    ys = np.linspace(0.08, 0.42, 300)
    xy = torch.as_tensor(np.stack([np.full(300, 0.005), ys], 1),
                         dtype=torch.float64)
    v = hjb_mod.eval_phi_at(phi, xy, t_new).numpy().ravel()
    return float(v.mean()), float(v.min()), int((v < 0).sum())


def main():
    os.makedirs(TMP, exist_ok=True)
    print(f"{'lam_0':>6} {'面积变化':>9} {'左边界φ均值':>11} {'最小':>9} {'侵蚀点数':>8}")
    for lam0 in LAM0_LIST:
        cfg = hjb_mod.HJBConfig(
            **BASE, lam_0=lam0,
            phi_out=os.path.join(TMP, f"phi_lam{lam0}.pt"),
            state_out=os.path.join(TMP, f"state_lam{lam0}.json"),
            fig_dir=os.path.join(TMP, f"fig_lam{lam0}"),
            seed=20260945 + 2,
        )
        hjb_mod.main(cfg)
        with open(cfg.state_out, encoding="utf-8") as f:
            st = json.load(f)
        da = st["area_after"] - st["area_before"]
        m, mn, n = edge_stat(cfg.phi_out, cfg.t_new)
        print(f"{lam0:>6} {da:>9.4f} {m:>11.4f} {mn:>9.4f} {n:>5}/300", flush=True)


if __name__ == "__main__":
    main()
