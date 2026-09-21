# -*- coding: utf-8 -*-
"""smoke_corner_freeze.py — 角部硬非设计域的快速冒烟测试（只做 1 步 HJB，不力学问解）。

验证：corner_freeze=0.08 时，角部 φ 在演化后保持为正（材料保留），
而对照组（corner_freeze=0）角部会被侵蚀。训练量刻意调小以求快。
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "start")
sys.path.insert(0, START_DIR)

import torch
import hjb_step_eik as hjb_mod

W = os.path.join(SCRIPT_DIR, "weights")


def corner_phi(phi_path, t_val, cfg):
    """测四角冻结区内 φ 的最小值（>0 则角部材料保留）。"""
    from networks import NetworkConfig, build_sdf_network
    phi = build_sdf_network(NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype))
    phi.load_state_dict(torch.load(phi_path, weights_only=True))
    phi.eval()
    cf = 0.08
    s = torch.linspace(0.0, cf, 33, dtype=cfg.dtype)
    X, Y = torch.meshgrid(s, s, indexing="xy")
    corners = []
    for cx, sx in [(0.0, 1.0), (cfg.lx, -1.0)]:
        for cy, sy in [(0.0, 1.0), (cfg.ly, -1.0)]:
            corners.append(torch.stack([cx + sx * X.reshape(-1),
                                        cy + sy * Y.reshape(-1)], dim=1))
    xy = torch.cat(corners, dim=0)
    with torch.no_grad():
        tt = torch.full((xy.shape[0], 1), t_val, dtype=cfg.dtype)
        pv = phi(torch.cat([xy, tt], dim=1)).squeeze(1)
    return float(pv.min()), float(pv.mean())


def run_one(cf_value, tag):
    cfg = hjb_mod.HJBConfig(
        t_current=0.0, dt=0.005,
        corner_freeze=cf_value,
        adam_steps=400, lbfgs_blocks=3,           # 快速冒烟（远小于正式量）
        pool_corner_freeze=2000, batch_corner_freeze=256, lbfgs_corner_freeze=256,
        phi_path=os.path.join(W, "phi_init.pt"),
        mech_dir=W, mech_suffix="_init.pt",
        phi_out=os.path.join(W, f"phi_smoke_{tag}.pt"),
        state_out=os.path.join(W, f"hjb_state_smoke_{tag}.json"),
        fig_dir=os.path.join(SCRIPT_DIR, "loop_figures", f"smoke_{tag}"),
        fem_check=False, quiet_figures=True,
    )
    hjb_mod.main(cfg)
    lo, mean = corner_phi(cfg.phi_out, cfg.t_new, cfg)
    print(f"[{tag}] corner_freeze={cf_value} | 演化后角部 φ：min={lo:+.4f}, mean={mean:+.4f}"
          f"（{'材料保留' if lo > -0.02 else '角部被侵蚀！'}）")
    return lo, mean


if __name__ == "__main__":
    print("=== 对照组：corner_freeze=0（关闭） ===")
    run_one(0.0, "off")
    print("\n=== 实验组：corner_freeze=0.08（开启） ===")
    run_one(0.08, "on")
