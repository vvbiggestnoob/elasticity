# -*- coding: utf-8 -*-
"""补跑 diag_lam0_sweep 的 lam_0=50 数据点（一次性诊断）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import diag_lam0_sweep as sweep
import hjb_step_eik as hjb_mod


def main():
    lam0 = 50.0
    cfg = hjb_mod.HJBConfig(
        **sweep.BASE, lam_0=lam0,
        phi_out=os.path.join(sweep.TMP, f"phi_lam{lam0}.pt"),
        state_out=os.path.join(sweep.TMP, f"state_lam{lam0}.json"),
        fig_dir=os.path.join(sweep.TMP, f"fig_lam{lam0}"),
        seed=20260945 + 2,
    )
    hjb_mod.main(cfg)
    m, mn, n = sweep.edge_stat(cfg.phi_out, cfg.t_new)
    print(f"EDGE {m:.6f} {mn:.6f} {n}/300", flush=True)


if __name__ == "__main__":
    main()
