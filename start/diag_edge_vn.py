# -*- coding: utf-8 -*-
"""diag_edge_vn.py — 四条边界上 V_n 的定量诊断。

回答的问题：给定某一步的几何与力学权重，在固定 α=1 与 Step D 自适应罚
（p = min(τω²/|ΔV|, p_cap)）两种罚压力下，四条边界上分别有多大比例的点
满足 V_n < 0（侵蚀条件）。左右边上 u=0（Dirichlet），V_n = ε:A:ε − p，
所以本质上是在比较各边应变能密度与罚压力的大小关系。

用法：
    python diag_edge_vn.py                          # 默认诊断 iter10 几何
    python diag_edge_vn.py phi_init.pt _init.pt 0.0 # 诊断初始几何
"""

import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from networks import NetworkConfig, build_mechanics_networks, build_sdf_network, MECH_NET_NAMES
import hjb_step_eik as hjb_mod


def main(phi_name: str = "phi_iter10.pt", mech_suffix: str = "_iter10.pt",
         t_slice: float = 0.5, tau: float = 1.1, p_cap: float = 1.0,
         v_target: float = 0.4):
    cfg = hjb_mod.HJBConfig(t_current=t_slice, v_target=v_target, fem_check=False)

    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly,
                            hidden_layers=cfg.mech_hidden_layers,
                            hidden_width=cfg.mech_hidden_width, dtype=cfg.dtype)
    mech = build_mechanics_networks(net_cfg)
    for name in MECH_NET_NAMES:
        mech[name].load_state_dict(torch.load(
            os.path.join(SCRIPT_DIR, "weights", name + mech_suffix), weights_only=True))
    mech.eval()
    mech.requires_grad_(False)

    phi = build_sdf_network(NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype))
    phi.load_state_dict(torch.load(
        os.path.join(SCRIPT_DIR, "weights", phi_name), weights_only=True))
    phi.eval()
    phi.requires_grad_(False)

    # 当前几何的 ω²（Rayleigh）与面积
    xy_int = hjb_mod.sample_interior_sobol(65536, cfg, seed=12345)
    omega2, area = hjb_mod.rayleigh_and_area(mech, phi, t_slice, xy_int, cfg)
    dV = area - v_target
    p_fixed = dV / 1.0                                   # 固定 α = 1
    p_adapt = min(tau * omega2 / max(abs(dV), 1e-12), p_cap)  # Step D 自适应
    print(f"几何：{phi_name} @ t={t_slice}，力学：*{mech_suffix}")
    print(f"ω²(Rayleigh) = {omega2:.6f} | area = {area:.4f} | ΔV = {dV:+.4f}")
    print(f"罚压力：固定 α=1 → p = {p_fixed:.4f} | 自适应 → p = {p_adapt:.4f}"
          f"（p_cap = {p_cap}）")
    print(f"（左右边侵蚀条件：p > ε:A:ε；上下边同理但动能项不恒零）")

    # 四条边上的点（含边中点剖面）
    n_v, n_h = 1000, 3200
    ys = torch.linspace(0.0, cfg.ly, n_v, dtype=cfg.dtype).unsqueeze(1)
    xs = torch.linspace(0.0, cfg.lx, n_h, dtype=cfg.dtype).unsqueeze(1)
    edges = {
        "left  (x=0)  ": torch.cat([torch.zeros_like(ys), ys], dim=1),
        "right (x=lx) ": torch.cat([torch.full_like(ys, cfg.lx), ys], dim=1),
        "bottom(y=0)  ": torch.cat([xs, torch.zeros_like(xs)], dim=1),
        "top   (y=ly) ": torch.cat([xs, torch.full_like(xs, cfg.ly)], dim=1),
    }

    print(f"\n{'边':<14} {'ε:A:ε mean/max':>18} {'ω²ρ|u|² max':>12} "
          f"{'V_n<0 比例(固定α)':>18} {'V_n<0 比例(自适应)':>18}")
    for ename, pts in edges.items():
        vn0, eng, kin = hjb_mod.compute_vn(mech, phi, pts, cfg, omega2, area)
        base = eng - omega2 * kin          # vn = base − penalty
        vn_fix = base - p_fixed
        vn_ada = base - p_adapt
        print(f"{ename} {float(eng.mean()):>8.3f}/{float(eng.max()):>8.3f} "
              f"{float((omega2 * kin).max()):>12.3f} "
              f"{float((vn_fix < 0).double().mean()):>18.1%} "
              f"{float((vn_ada < 0).double().mean()):>18.1%}")

    # 左边界的 y 向剖面（看中段 vs 近角的对比）
    print("\n左边界剖面（y, ε:A:ε, ω²ρ|u|², V_n 固定α, V_n 自适应）：")
    for yv in (0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.45, 0.48):
        pt = torch.tensor([[0.0, yv]], dtype=cfg.dtype)
        vn0, eng, kin = hjb_mod.compute_vn(mech, phi, pt, cfg, omega2, area)
        base = float(eng - omega2 * kin)
        print(f"  y={yv:.2f} | {float(eng):8.3f} | {float(omega2 * kin):8.3f} "
              f"| {base - p_fixed:+8.3f} | {base - p_adapt:+8.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args[:1], **({} if len(args) <= 1 else {"mech_suffix": args[1]}),
         **({} if len(args) <= 2 else {"t_slice": float(args[2])}))
