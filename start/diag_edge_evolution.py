# -*- coding: utf-8 -*-
"""诊断：新罚参数运行下，左/右边界为什么不被侵蚀。

针对 exp01_bridge 当前运行的前若干步，逐步重建 V_n 的三个分量
（应变能密度 se、动能密度 ω²ρ|u|²、罚压力 p）在左右边界附近的分布，
并跟踪零水平集在边界上的位置变化。数据存 diag_figures/edge_evolution.json，
图存 diag_figures/edge_evolution.png。

用法：python diag_edge_evolution.py [k_max]   （k_max 默认 3，分析第 1..k_max 步）
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import hjb_step_eik as hjb_mod
from networks import NetworkConfig, build_sdf_network

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(SCRIPT_DIR, "..", "exp01_bridge", "weights")
OUT_JSON = os.path.join(SCRIPT_DIR, "diag_figures", "edge_evolution.json")
OUT_FIG = os.path.join(SCRIPT_DIR, "diag_figures", "edge_evolution.png")

DT = 0.005
CORNER_FREEZE = 0.08


def load_phi(k: int):
    phi = build_sdf_network(NetworkConfig(dtype=torch.float64))
    name = "phi_init.pt" if k == 0 else f"phi_iter{k}.pt"
    phi.load_state_dict(torch.load(os.path.join(WEIGHTS, name), weights_only=True))
    phi.eval().requires_grad_(False)
    return phi


def load_mech(k: int):
    cfg = hjb_mod.HJBConfig(dtype=torch.float64)
    net_cfg = NetworkConfig(hidden_layers=cfg.mech_hidden_layers,
                            hidden_width=cfg.mech_hidden_width, dtype=torch.float64)
    mech = hjb_mod.build_mechanics_networks(net_cfg)
    suffix = "_init.pt" if k == 0 else f"_iter{k}.pt"
    for n in ("u_x", "u_y"):
        mech[n].load_state_dict(torch.load(os.path.join(WEIGHTS, n + suffix),
                                           weights_only=True))
    mech.eval().requires_grad_(False)
    return mech


def vn_at(k: int, xy: np.ndarray):
    """第 k 步实际使用的 V_n 及其分量。xy: (N,2) numpy。
    返回 (vn, se, ke, p, state)。"""
    with open(os.path.join(WEIGHTS, f"hjb_state_iter{k}.json"), encoding="utf-8") as f:
        st = json.load(f)
    cfg = hjb_mod.HJBConfig(t_current=st["t_current"], alpha=st["alpha"],
                            corner_freeze=CORNER_FREEZE, dtype=torch.float64)
    phi_ref = load_phi(k - 1)
    mech = load_mech(k - 1)
    xt = torch.as_tensor(xy, dtype=torch.float64)
    vn, eng, kin = hjb_mod.compute_vn(mech, phi_ref, xt, cfg,
                                      st["omega2_before"], st["area_before"])
    p = (st["area_before"] - 0.4) / st["alpha"]
    return vn.numpy().ravel(), eng.numpy().ravel(), kin.numpy().ravel(), p, st


def phi_on_line(k: int, xy: np.ndarray) -> np.ndarray:
    """几何 k 在其窗口右端切片（t = k·dt）的 φ 值。"""
    phi = load_phi(k)
    xt = torch.as_tensor(xy, dtype=torch.float64)
    return hjb_mod.eval_phi_at(phi, xt, k * DT).numpy().ravel()


def main(k_max: int = 3):
    ny = 501
    ys = np.linspace(0.0, 0.5, ny)
    out = {"ys": ys.tolist(), "steps": {}}

    # --- 每条边界、每一步：V_n 分解（在 x=0.005 的贴边线上，避开角部冻结区）---
    for side, x in [("left", 0.005), ("right", 1.595)]:
        xy = np.stack([np.full(ny, x), ys], axis=1)
        for k in range(1, k_max + 1):
            vn, se, ke, p, st = vn_at(k, xy)
            w2 = st["omega2_before"]
            rec = out["steps"].setdefault(f"iter{k}", {})
            rec[f"{side}_vn"] = vn.tolist()
            rec[f"{side}_freq"] = (se - w2 * ke).tolist()
            rec["p"] = p
            rec["alpha"] = st["alpha"]
            rec["omega2_before"] = w2
            rec["area_before"] = st["area_before"]

    # --- 零水平集在边界上的位置：φ(x=0.005, y) 逐步 ---
    for k in range(0, k_max + 1):
        for side, x in [("left", 0.005), ("right", 1.595)]:
            xy = np.stack([np.full(ny, x), ys], axis=1)
            out["steps"].setdefault(f"phi{k}", {})[f"{side}_phi"] = \
                phi_on_line(k, xy).tolist()

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f)

    # --- 汇总打印 ---
    live = (ys > CORNER_FREEZE) & (ys < 0.5 - CORNER_FREEZE)   # 角部冻结区以外
    print("=" * 78)
    print("左边界中段（x=0.005，y∈[0.08,0.42]，角部冻结区以外）逐步 V_n 分解：")
    print(f"{'步':>3} {'se−ke 均值':>10} {'se−ke 最小':>10} {'罚 p':>7} "
          f"{'V_n 均值':>9} {'V_n 最小':>9} {'V_n<0 比例':>10}")
    for k in range(1, k_max + 1):
        rec = out["steps"][f"iter{k}"]
        freq = np.array(rec["left_freq"])[live]
        vn = np.array(rec["left_vn"])[live]
        print(f"{k:>3} {freq.mean():>10.4f} {freq.min():>10.4f} {rec['p']:>7.4f} "
              f"{vn.mean():>9.4f} {vn.min():>9.4f} {np.mean(vn < 0):>10.1%}")

    print("\n零水平集位置（左边界 x=0.005 处 φ 的符号变化，负=该处已侵蚀）：")
    for k in range(0, k_max + 1):
        v = np.array(out["steps"][f"phi{k}"]["left_phi"])
        eroded = ys[(v < 0) & live]
        desc = (f"y∈[{eroded.min():.3f},{eroded.max():.3f}] 共 {len(eroded)} 点"
                if len(eroded) else "无侵蚀")
        print(f"  phi_{k} @ t={k * DT:.3f}：{desc}，φ 最小值 {v[live].min():+.4f}")

    # --- 图：左边 V_n 分解逐步剖面 + φ 逐步剖面 ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
    ax = axes[0]
    for k in range(1, k_max + 1):
        rec = out["steps"][f"iter{k}"]
        ax.plot(np.array(rec["left_freq"]), ys, label=f"step {k}: se−ke")
    for k in range(1, k_max + 1):
        rec = out["steps"][f"iter{k}"]
        ax.axvline(rec["p"], ls="--", alpha=0.5,
                   label=f"step {k}: p = {rec['p']:.3f}")
    ax.set_xlabel("值")
    ax.set_ylabel("y")
    ax.set_title("左边界：频率项 vs 罚压力（se−ke > p 才会长材料）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for k in range(1, k_max + 1):
        vn = np.array(out["steps"][f"iter{k}"]["left_vn"])
        ax.plot(vn, ys, label=f"step {k}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("V_n")
    ax.set_title("左边界：总速度 V_n（<0 = 侵蚀）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for k in range(0, k_max + 1):
        v = np.array(out["steps"][f"phi{k}"]["left_phi"])
        ax.plot(v, ys, label=f"phi_{k}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("φ(x=0.005, y)")
    ax.set_title("左边界：SDF 逐步剖面（<0 = 空洞）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for a in axes:
        a.axhspan(0, CORNER_FREEZE, color="gray", alpha=0.15)
        a.axhspan(0.5 - CORNER_FREEZE, 0.5, color="gray", alpha=0.15)
    fig.suptitle("左/右边界侵蚀诊断（灰带 = 角部冻结区）")
    fig.savefig(OUT_FIG, dpi=140)
    plt.close(fig)
    print(f"\n[保存] 数据 -> {OUT_JSON}")
    print(f"[保存] 图 -> {OUT_FIG}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
