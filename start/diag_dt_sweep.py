# -*- coding: utf-8 -*-
"""实测：放大 dt 能否提高每步兑现的侵蚀量（lam_0=1.0 修复后）。

固定重跑第 2 步（phi_1 → phi_2'），只改 dt（窗口 [0.005, 0.005+dt]），
用运行的完整训练量（adam 1500 / lbfgs 100），报告面积变化与左边界 φ。
对照：理论处方 ≈ −0.0084·(dt/0.005)；dt=0.005 实测 −0.0013（兑现 15%）。

用法：python diag_dt_sweep.py
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
TMP = os.path.join(SCRIPT_DIR, "diag_figures", "dt_sweep_tmp")

BASE = dict(
    t_current=0.005, alpha=0.562622, v_target=0.4,
    corner_freeze=0.08, w_corner_freeze=10.0, lam_0=1.0,
    phi_path=os.path.join(WEIGHTS, "phi_iter1.pt"),
    mech_dir=WEIGHTS, mech_suffix="_iter1.pt",
    adam_steps=800, lbfgs_blocks=30,
    fem_check=False, quiet_figures=True,
    dtype=torch.float64,
)

DT_LIST = [0.005, 0.01, 0.02]


def edge_and_area(phi_path: str, t_new: float):
    phi = build_sdf_network(NetworkConfig(dtype=torch.float64))
    phi.load_state_dict(torch.load(phi_path, weights_only=True))
    phi.eval().requires_grad_(False)
    ys = np.linspace(0.08, 0.42, 300)
    xy = torch.as_tensor(np.stack([np.full(300, 0.005), ys], 1),
                         dtype=torch.float64)
    v = hjb_mod.eval_phi_at(phi, xy, t_new).numpy().ravel()
    return float(v.mean()), float(v.min())


def main():
    os.makedirs(TMP, exist_ok=True)
    print(f"{'dt':>6} {'面积变化':>9} {'兑现率':>7} {'左边界φ均值':>11} {'最小':>8}")
    for dt in DT_LIST:
        cfg = hjb_mod.HJBConfig(
            **BASE, dt=dt,
            phi_out=os.path.join(TMP, f"phi_dt{dt}.pt"),
            state_out=os.path.join(TMP, f"state_dt{dt}.json"),
            fig_dir=os.path.join(TMP, f"fig_dt{dt}"),
            seed=20260945 + 2,
        )
        hjb_mod.main(cfg)
        with open(cfg.state_out, encoding="utf-8") as f:
            st = json.load(f)
        da = st["area_after"] - st["area_before"]
        theory = -0.0084 * (dt / 0.005)
        m, mn = edge_and_area(cfg.phi_out, cfg.t_new)
        print(f"{dt:>6} {da:>9.4f} {da / theory:>6.0%} {m:>11.4f} {mn:>8.4f}",
              flush=True)


if __name__ == "__main__":
    main()
