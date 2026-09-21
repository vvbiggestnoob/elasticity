# -*- coding: utf-8 -*-
"""补跑 diag_lam0_sweep 的中间 lam_0 点（找边界侵蚀峰值）。用法: python diag_lam0_peak.py 5 7"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diag_lam0_sweep as sweep
import hjb_step_eik as hjb_mod


def main():
    for arg in sys.argv[1:]:
        lam0 = float(arg)
        cfg = hjb_step_eik_cfg = hjb_mod.HJBConfig(
            **sweep.BASE, lam_0=lam0,
            phi_out=os.path.join(sweep.TMP, f"phi_lam{lam0}.pt"),
            state_out=os.path.join(sweep.TMP, f"state_lam{lam0}.json"),
            fig_dir=os.path.join(sweep.TMP, f"fig_lam{lam0}"),
            seed=20260945 + 2,
        )
        hjb_mod.main(cfg)
        print(f"DONE lam_0={lam0}", flush=True)


if __name__ == "__main__":
    main()
