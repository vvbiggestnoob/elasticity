# -*- coding: utf-8 -*-
"""mech_init_v2.py — 角部加密实验版力学初始化（自然演变路线，不加硬非设计域）

背景（2026-09-14 FEM 对拍发现）：PINN 力学网络在四角（左右固支边与上下自由边
相交的应力奇异点）把应变能密度低估约 3 倍（≈1.2 vs FEM ≈3.3），其余位置吻合
（左右边中段 0.9、上下边中段 0.97、重块动能项 1.0、ω² 相对误差 0.4%）。
角部低估使 HJB 演化的角部保护窗口（p_cap < 角部应变能）失效，角部被误侵蚀。

本实验只改采样、不改网络结构与损失形式：
  新增四角加密点池（corner_side 见方的角域，pde+cons 残差，权重 w_corner），
  Adam 点池 / L-BFGS 固定集 / 验证集同步加入角点；其余与 mech_init.py 一致。
默认从既有 weights/*_init.pt 热启动做角部精调（init_dir 可指别的目录），
输出到 weights_v2/ 与 mech_figures_v2/，不覆盖既有产物。

对照评估：python diag_fem_vs_pinn_corners.py weights_v2 _init.pt
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

import mech_init
from mech_init import (MechInitConfig, material_fields, mech_quantities,
                       constitutive_stress, block_losses, dirichlet_loss,
                       neumann_loss, sample_interior_pool, sample_block_pool,
                       sample_boundary_pools, minibatch, save_field_figure,
                       save_loss_figure, diagnose, _resolve)
from networks import NetworkConfig, build_mechanics_networks, MECH_NET_NAMES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 配置（继承 mech_init，只加角部相关项；默认热启动 + 较小训练量做角部精调）
# ---------------------------------------------------------------------------

@dataclass
class MechInitV2Config(MechInitConfig):
    # 四角加密：corner_side 见方的角域（覆盖角部探针半径 0.05 与 4β 带）
    corner_side: float = 0.06
    pool_corner: int = 8000          # Adam 角部点池
    batch_corner: int = 1024         # 每步 minibatch 角点数
    lbfgs_corner: int = 2000         # L-BFGS 固定集角点数
    val_corner: int = 1024           # 验证集角点数
    w_corner: float = 1.0            # 角部 (pde+cons) 损失权重

    # 热启动来源目录（None = 从 weights_dir 内读，同 mech_init 行为）
    init_dir: Optional[str] = "weights"

    # v2 默认：热启动角部精调（较小训练量；从头训练把 init_suffix=None、
    # adam_steps=20000、lbfgs_blocks=160 即可恢复完整训练）
    adam_steps: int = 4000
    lbfgs_blocks: int = 60
    lr_init: float = 2e-4            # 热启动精调用更小的 lr，保护已学好的全场
    init_suffix: Optional[str] = "_init.pt"

    # 独立输出，不覆盖 v1 产物
    weights_dir: str = "weights_v2"
    fig_dir: str = "mech_figures_v2"


# ---------------------------------------------------------------------------
# 角部采样
# ---------------------------------------------------------------------------

def sample_corner_pool(n: int, cfg: MechInitV2Config, gen: torch.Generator) -> torch.Tensor:
    """四角加密点池：四个 corner_side 见方的角域内均匀采样（最大余数法分配）。"""
    base, rem = divmod(n, 4)
    counts = [base + (1 if i < rem else 0) for i in range(4)]
    # (角点 x, 角点 y, x 向域内符号, y 向域内符号)
    corners = [(0.0, 0.0, 1.0, 1.0), (cfg.lx, 0.0, -1.0, 1.0),
               (0.0, cfg.ly, 1.0, -1.0), (cfg.lx, cfg.ly, -1.0, -1.0)]
    pts = []
    for (cx, cy, sx, sy), m in zip(corners, counts):
        if m == 0:
            continue
        u = torch.rand(m, 2, generator=gen, dtype=cfg.dtype) * cfg.corner_side
        pts.append(torch.stack([cx + sx * u[:, 0], cy + sy * u[:, 1]], dim=1))
    return torch.cat(pts, dim=0)


# ---------------------------------------------------------------------------
# 总损失（v1 七项 + 角部项）
# ---------------------------------------------------------------------------

def compute_total_loss_v2(mech, phi_net, pts, cfg: MechInitV2Config,
                          use_ray: bool, create_graph: bool = True):
    """v1 总损失 + w_corner·(角部 pde+cons)。pts 需多一个 "corner" 键。"""
    il = mech_init.interior_losses(mech, phi_net, pts["interior"], cfg, create_graph)
    l_pde_blk, l_cons_blk = block_losses(mech, phi_net, pts["block"], cfg, create_graph)
    l_pde_cor, l_cons_cor = block_losses(mech, phi_net, pts["corner"], cfg, create_graph)
    l_dir = dirichlet_loss(mech, phi_net, pts["dir"], cfg)
    l_neu = neumann_loss(mech, phi_net, pts["neu"], cfg)

    l_norm = (il.m - 1.0) ** 2
    l_ray = (il.r_ray / cfg.omega2_fem - 1.0) ** 2

    total = (
        cfg.w_pde * il.l_pde
        + cfg.w_cons * il.l_cons
        + cfg.w_blk * (l_pde_blk + l_cons_blk)
        + cfg.w_corner * (l_pde_cor + l_cons_cor)
        + cfg.w_dir * l_dir
        + cfg.w_neu * l_neu
        + cfg.w_norm * l_norm
    )
    if use_ray:
        total = total + cfg.w_ray * l_ray

    parts = {
        "total": float(total.detach()),
        "pde": float(il.l_pde.detach()),
        "cons": float(il.l_cons.detach()),
        "blk": float((l_pde_blk + l_cons_blk).detach()),
        "cor": float((l_pde_cor + l_cons_cor).detach()),
        "dir": float(l_dir.detach()),
        "neu": float(l_neu.detach()),
        "norm": float(l_norm.detach()),
        "ray": float(l_ray.detach()) if use_ray else float("nan"),
        "m": float(il.m.detach()),
        "R": float(il.r_ray.detach()),
    }
    return total, parts


def evaluate_loss_v2(mech, phi_net, pts, cfg: MechInitV2Config) -> float:
    total, _ = compute_total_loss_v2(mech, phi_net, pts, cfg,
                                     use_ray=True, create_graph=False)
    return float(total.detach())


# ---------------------------------------------------------------------------
# 训练（结构同 mech_init，点集多 "corner" 键）
# ---------------------------------------------------------------------------

def train_adam_v2(mech, phi_net, pools, cfg: MechInitV2Config,
                  gen: torch.Generator, history: list):
    if cfg.adam_steps <= 0:
        print("[Adam] adam_steps=0，跳过 Adam 段，直接进入 L-BFGS")
        return
    params = list(mech.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr_init)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.adam_steps, eta_min=cfg.lr_final)

    print(f"[Adam] 开始：{cfg.adam_steps} 步，lr {cfg.lr_init}→{cfg.lr_final}（余弦），"
          f"每步 batch {cfg.batch_interior}+{cfg.batch_block}+{cfg.batch_corner}(角)"
          f"+2×{cfg.batch_boundary_per_edge}")
    t0 = time.time()
    for step in range(cfg.adam_steps):
        pts = {
            "interior": minibatch(pools["interior"], cfg.batch_interior, gen),
            "block": minibatch(pools["block"], cfg.batch_block, gen),
            "corner": minibatch(pools["corner"], cfg.batch_corner, gen),
            "dir": minibatch(pools["dir"], 2 * cfg.batch_boundary_per_edge, gen),
            "neu": minibatch(pools["neu"], 2 * cfg.batch_boundary_per_edge, gen),
        }
        use_ray = step >= cfg.rayleigh_warmup
        total, parts = compute_total_loss_v2(mech, phi_net, pts, cfg, use_ray=use_ray)

        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == cfg.adam_steps - 1:
            parts["step"] = step
            history.append(parts)
            print(f"[Adam] step {step:5d} | total {parts['total']:.4e} | "
                  f"pde {parts['pde']:.3e} cons {parts['cons']:.3e} "
                  f"blk {parts['blk']:.3e} cor {parts['cor']:.3e} | "
                  f"dir {parts['dir']:.3e} neu {parts['neu']:.3e} "
                  f"norm {parts['norm']:.3e} ray {parts['ray']:.3e} | "
                  f"m {parts['m']:.4f} R {parts['R']:.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.0f}s")

        if not cfg.quiet_figures and (step % cfg.fig_every == 0
                                      or step == cfg.adam_steps - 1):
            save_field_figure(mech, phi_net, cfg,
                              os.path.join(cfg.fig_dir, f"adam_step{step:06d}.png"),
                              title=f"Adam step {step}")
    print(f"[Adam] 结束，用时 {time.time()-t0:.0f}s")


def train_lbfgs_v2(mech, phi_net, cfg: MechInitV2Config, history: list):
    import copy
    import math

    params = list(mech.parameters())
    gen_train = torch.Generator().manual_seed(cfg.seed + 100)
    gen_val = torch.Generator().manual_seed(cfg.seed + 200)

    train_pts = {
        "interior": sample_interior_pool(cfg.lbfgs_interior, cfg, cfg.seed + 101),
        "block": sample_block_pool(cfg.lbfgs_block, cfg, gen_train),
        "corner": sample_corner_pool(cfg.lbfgs_corner, cfg, gen_train),
        "dir": None, "neu": None,
    }
    train_pts["dir"], train_pts["neu"] = sample_boundary_pools(
        cfg.lbfgs_boundary_per_edge, cfg, gen_train)

    val_pts = {
        "interior": sample_interior_pool(cfg.val_interior, cfg, cfg.seed + 201),
        "block": sample_block_pool(cfg.val_block, cfg, gen_val),
        "corner": sample_corner_pool(cfg.val_corner, cfg, gen_val),
        "dir": None, "neu": None,
    }
    val_pts["dir"], val_pts["neu"] = sample_boundary_pools(
        cfg.val_boundary_per_edge, cfg, gen_val)

    best_val = math.inf
    best_state = copy.deepcopy(mech.state_dict())
    prev_metric: Optional[float] = None
    stall = 0

    print(f"[L-BFGS] 开始：≤{cfg.lbfgs_blocks} 块 × {cfg.lbfgs_max_iter} 次，强 Wolfe，"
          f"固定点集 {cfg.lbfgs_interior}+{cfg.lbfgs_block}+{cfg.lbfgs_corner}(角)"
          f"+2×{cfg.lbfgs_boundary_per_edge}")
    opt = torch.optim.LBFGS(
        params, max_iter=cfg.lbfgs_max_iter, history_size=cfg.lbfgs_history,
        line_search_fn="strong_wolfe", tolerance_grad=1e-16, tolerance_change=1e-16)

    def closure():
        opt.zero_grad()
        total, _ = compute_total_loss_v2(mech, phi_net, train_pts, cfg,
                                         use_ray=True, create_graph=True)
        total.backward()
        return total

    t0 = time.time()
    for block in range(cfg.lbfgs_blocks):
        opt.step(closure)

        train_loss = evaluate_loss_v2(mech, phi_net, train_pts, cfg)
        val_loss = evaluate_loss_v2(mech, phi_net, val_pts, cfg)

        rel_improve = (best_val - val_loss) / max(abs(best_val), 1e-30)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            if cfg.lbfgs_rollback:
                best_state = copy.deepcopy(mech.state_dict())
        metric = train_loss if cfg.early_stop_metric == "train" else val_loss
        drop_rel = (float("inf") if prev_metric is None
                    else (prev_metric - metric) / max(abs(prev_metric), 1e-30))
        prev_metric = metric
        stall = stall + 1 if drop_rel < cfg.early_stop_rtol else 0

        history.append({"step": f"lbfgs_{block}", "total": train_loss, "val": val_loss})
        mark = " *best" if is_best else ""
        print(f"[L-BFGS] block {block:3d} | train {train_loss:.6e} | "
              f"val {val_loss:.6e} | rel_improve {rel_improve:+.2e}{mark} | "
              f"{time.time()-t0:.0f}s")

        if not cfg.quiet_figures and (block % cfg.fig_every_blocks == 0
                                      or block == cfg.lbfgs_blocks - 1):
            save_field_figure(mech, phi_net, cfg,
                              os.path.join(cfg.fig_dir, f"lbfgs_block{block:03d}.png"),
                              title=f"L-BFGS block {block}")

        if cfg.lbfgs_early_stop and stall >= cfg.early_stop_patience:
            print(f"[L-BFGS] 早停：连续 {stall} 块 {cfg.early_stop_metric} "
                  f"损失相对下降 < {cfg.early_stop_rtol}")
            break

    if cfg.lbfgs_rollback:
        mech.load_state_dict(best_state)
        print(f"[L-BFGS] 结束，回滚到验证最优（best val = {best_val:.6e}），"
              f"用时 {time.time()-t0:.0f}s")
    else:
        print(f"[L-BFGS] 结束，保留最终权重（best val = {best_val:.6e} 仅供参考），"
              f"用时 {time.time()-t0:.0f}s")
    return best_val


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(cfg: Optional[MechInitV2Config] = None):
    cfg = cfg or MechInitV2Config()
    cfg.weights_dir = _resolve(cfg.weights_dir)
    cfg.fig_dir = _resolve(cfg.fig_dir)
    cfg.phi_path = _resolve(cfg.phi_path)
    if cfg.init_dir is not None:
        cfg.init_dir = _resolve(cfg.init_dir)
    os.makedirs(cfg.weights_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    phi_net = mech_init.load_frozen_phi(cfg)
    print(f"已加载冻结 SDF：{cfg.phi_path}")

    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly,
                            hidden_layers=cfg.hidden_layers,
                            hidden_width=cfg.hidden_width, dtype=cfg.dtype)
    mech = build_mechanics_networks(net_cfg)
    print(f"力学网络：{list(mech.keys())}（结构与 v1 一致）")

    if cfg.init_suffix is not None:
        src_dir = cfg.init_dir or cfg.weights_dir
        for name in MECH_NET_NAMES:
            p = os.path.join(src_dir, f"{name}{cfg.init_suffix}")
            mech[name].load_state_dict(torch.load(p, weights_only=True))
        print(f"力学网络热启动：{src_dir}/*{cfg.init_suffix}")

    if not cfg.quiet_figures:
        save_field_figure(mech, phi_net, cfg,
                          os.path.join(cfg.fig_dir, "step000000_initial.png"),
                          title="initial (warm-started)")

    gen = torch.Generator().manual_seed(cfg.seed + 1)
    pools = {
        "interior": sample_interior_pool(cfg.pool_interior, cfg, cfg.seed + 2),
        "block": sample_block_pool(cfg.pool_block, cfg, gen),
        "corner": sample_corner_pool(cfg.pool_corner, cfg, gen),
        "dir": None, "neu": None,
    }
    pools["dir"], pools["neu"] = sample_boundary_pools(
        cfg.pool_boundary_per_edge, cfg, gen)
    print(f"Adam 点池：interior {cfg.pool_interior}，block {cfg.pool_block}，"
          f"corner {cfg.pool_corner}（{cfg.corner_side} 见方 ×4），"
          f"dir/neu 各 2×{cfg.pool_boundary_per_edge}")

    adam_history: list = []
    train_adam_v2(mech, phi_net, pools, cfg, gen, adam_history)

    lbfgs_history: list = []
    train_lbfgs_v2(mech, phi_net, cfg, lbfgs_history)

    save_loss_figure(adam_history, lbfgs_history, cfg,
                     os.path.join(cfg.fig_dir, "loss_history.png"))
    save_field_figure(mech, phi_net, cfg,
                      os.path.join(cfg.fig_dir, "final.png"), title="final (v2)")

    for name in MECH_NET_NAMES:
        torch.save(mech[name].state_dict(),
                   os.path.join(cfg.weights_dir, f"{name}{cfg.save_suffix}"))
    print(f"已保存 v2 力学网络权重 -> {cfg.weights_dir}/*{cfg.save_suffix}")

    diag = diagnose(mech, phi_net, cfg)
    print("力学初始化 v2（角部加密实验）完成。")
    return diag


if __name__ == "__main__":
    main()
