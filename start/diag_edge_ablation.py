# -*- coding: utf-8 -*-
"""消融实验：定位左/右边界 φ 系统性膨胀（回填）的来源。

固定重跑第 2 步（phi_1 → phi_2'，t∈[0.005,0.01]），只改损失权重：
  base   : 与运行一致 lam_0=0.1, lam_anchor=0.5, lam_eik=0.001, lam_r=1.0
  init10 : lam_0=1.0（初始条件 ×10）
  noeik  : lam_eik=0（关 eikonal）
  anch10 : lam_anchor=5.0（制造解锚点 ×10）
  noanchor: lam_anchor=0（关制造解锚点）
每组输出到临时目录（不污染运行权重），报告左边界中段 φ(t_2) 的均值/最小值。
处方：制造解目标 mean ≈ +0.0082，min ≈ −0.0029（侵蚀）。实际 base 约 +0.019/+0.013（回填）。

用法：python diag_edge_ablation.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

import hjb_step_eik as hjb_mod

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(SCRIPT_DIR, "..", "exp01_bridge", "weights")
TMP = os.path.join(SCRIPT_DIR, "diag_figures", "edge_ablation_tmp")

# 与运行一致的第 2 步配置（训练量适当压缩以省时；fem_check 关闭）
BASE = dict(
    t_current=0.005, dt=0.005, alpha=0.562622, v_target=0.4,
    corner_freeze=0.08, w_corner_freeze=10.0,
    phi_path=os.path.join(WEIGHTS, "phi_iter1.pt"),
    mech_dir=WEIGHTS, mech_suffix="_iter1.pt",
    adam_steps=800, lbfgs_blocks=30,
    fem_check=False, quiet_figures=True,
    dtype=torch.float64,
)

VARIANTS = [
    ("base", {}),
    ("init10", {"lam_0": 1.0}),
    ("noeik", {"lam_eik": 0.0}),
    ("anch10", {"lam_anchor": 5.0}),
    ("noanchor", {"lam_anchor": 0.0}),
]


def edge_stat(phi_path: str):
    phi = hjb_mod.build_sdf_network(
        __import__("networks").NetworkConfig(dtype=torch.float64))
    phi.load_state_dict(torch.load(phi_path, weights_only=True))
    phi.eval().requires_grad_(False)
    ys = np.linspace(0.08, 0.42, 300)
    xy = torch.as_tensor(np.stack([np.full(300, 0.005), ys], 1),
                         dtype=torch.float64)
    v = hjb_mod.eval_phi_at(phi, xy, 0.010).numpy().ravel()
    return float(v.mean()), float(v.min())


def main():
    os.makedirs(TMP, exist_ok=True)
    print(f"{'变体':>9} {'左边界 φ(t2) 均值':>16} {'最小值':>9}   （处方：mean≈+0.008, min≈−0.003；回填=偏大）")
    for name, over in VARIANTS:
        cfg = hjb_mod.HJBConfig(
            **BASE, **over,
            phi_out=os.path.join(TMP, f"phi_{name}.pt"),
            state_out=os.path.join(TMP, f"state_{name}.json"),
            fig_dir=os.path.join(TMP, f"fig_{name}"),
            seed=20260945 + 2,
        )
        hjb_mod.main(cfg)
        m, mn = edge_stat(cfg.phi_out)
        print(f"{name:>9} {m:>16.4f} {mn:>9.4f}", flush=True)


if __name__ == "__main__":
    main()
