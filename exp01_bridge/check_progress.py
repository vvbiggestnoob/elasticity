# -*- coding: utf-8 -*-
"""check_progress.py — exp01_bridge 进度监控：角部是否保住 + 体积/拓扑演变。

读 weights/loop_history.json 与各 phi_iter{k}.pt，输出：
  - 体积、ω²_FEM 轨迹；
  - 四角冻结区内 φ 最小值（>0 = 角部材料保住）；
  - 左/右边中段与角部的 φ 对比（看侵蚀顺序是否与上一run相反）。
"""
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "start")
sys.path.insert(0, START_DIR)

from networks import NetworkConfig, build_sdf_network
import hjb_step_eik as hjb_mod

W = os.path.join(SCRIPT_DIR, "weights")
DT = 0.005


def phi_at(phi, xy, t_val, dtype):
    with torch.no_grad():
        tt = torch.full((xy.shape[0], 1), t_val, dtype=dtype)
        return phi(torch.cat([xy, tt], dim=1)).squeeze(1)


def main():
    import json
    import glob
    cfg = hjb_mod.HJBConfig(t_current=0.0, fem_check=False)
    dtype = cfg.dtype
    phi = build_sdf_network(NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=dtype))

    # 探针：四角冻结区内一点 + 左右边中段 + 上下边中段
    probes = {
        "左下角内(.03,.03)": (0.03, 0.03), "右下角内(1.57,.03)": (1.57, 0.03),
        "左上角内(.03,.47)": (0.03, 0.47), "右上角内(1.57,.47)": (1.57, 0.47),
        "左边中(0,.25)": (0.0, 0.25), "右边中(1.6,.25)": (1.6, 0.25),
        "下边中(.4,0)": (0.4, 0.0), "上边中(.4,.5)": (0.4, 0.5),
    }
    names = list(probes)
    xy = torch.tensor([probes[n] for n in names], dtype=dtype)

    # 历史（体积 / ω²）
    hist_path = os.path.join(W, "loop_history.json")
    hist = []
    if os.path.exists(hist_path):
        with open(hist_path, encoding="utf-8") as f:
            hist = json.load(f)["history"]

    files = sorted(glob.glob(os.path.join(W, "phi_iter*.pt")),
                   key=lambda p: int("".join(c for c in os.path.basename(p)
                                             if c.isdigit())))
    ks = [int("".join(c for c in os.path.basename(f) if c.isdigit())) for f in files]
    if not ks:
        print("尚无 phi_iter{k}.pt 产出"); return

    print(f"{'k':>3} {'面积':>7} {'ω²FEM':>7} | " + " ".join(f"{n[:6]:>7}" for n in names))
    area_map = {r["iter"]: r.get("area_after") for r in hist}
    om_map = {r["iter"]: r.get("omega2_fem_after") for r in hist}
    for k, f in zip(ks, files):
        phi.load_state_dict(torch.load(f, weights_only=True))
        phi.eval()
        pv = phi_at(phi, xy, k * DT, dtype)
        a = area_map.get(k); o = om_map.get(k)
        a_s = f"{a:.4f}" if a is not None else "  —  "
        o_s = f"{o:.4f}" if o is not None else "  —  "
        print(f"{k:>3} {a_s:>7} {o_s:>7} | " + " ".join(f"{float(v):+7.3f}" for v in pv))

    # 判定：四角是否保住（φ>0），边中段是否侵蚀（φ<0）
    f_last = files[-1]
    phi.load_state_dict(torch.load(f_last, weights_only=True))
    phi.eval()
    pv = phi_at(phi, xy, ks[-1] * DT, dtype)
    cor_min = float(pv[:4].min())
    print(f"\n最新 k={ks[-1]}：四角冻结区 φ 最小值 = {cor_min:+.4f}"
          f"（{'✓ 角部保住' if cor_min > -0.02 else '✗ 角部仍被侵蚀'}）")


if __name__ == "__main__":
    main()
