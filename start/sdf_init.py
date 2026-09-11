# -*- coding: utf-8 -*-
"""
SDF 初始化实现（对应《SDF初始化说明.md》，唯一权威说明；数值取自《实验设置与计算范围.md》）

目标：把 SDF 网络 phi(x, y, t) 在 t = 0 切片上有监督回归拟合到全材料距离函数
    phi0(x, y) = min(x, Lx - x, y, Ly - y)        （offset = 0，零水平集压在域边界上）

损失：L = 内部 MSE + w_bnd * 边界带 MSE          （w_bnd = 10）
  - 内部点：Adam 阶段每步均匀重采样；L-BFGS 阶段固定 Sobol 点集
  - 边界带点：沿四边内侧窄带（带宽 4β = 0.04）采样，按边长比例分配，
    每边内隔点严格压在边上（d = 0），外加 4 个角点

训练流程（两阶段）：
  阶段 A：Adam，lr = 2e-3，5000 步，每步重采样 2048 内部点 + 1024 边界带点
  阶段 B：L-BFGS 精调，固定点集（16384 Sobol 内部点 + 4096 边界带点），
          20 块 x 70 次内部迭代，强 Wolfe 线搜索；独立验证集（8192 + 2048）只评估不训练，
          连续若干块相对改善低于阈值则早停，结束后回滚到验证损失最优的权重

全程 float64。训练完成后保存权重 weights/phi_init.pt，并做诊断自检（拟合误差、
|grad phi| 统计、边界上的 S 值）。

日志：每 100 步打印一次损失，每 500 步保存一次图片（L-BFGS 阶段按块打印/绘图）。
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.quasirandom import SobolEngine

import matplotlib

matplotlib.use("Agg")  # 无界面后端，只保存图片
import matplotlib.pyplot as plt

from networks import NetworkConfig, build_sdf_network

_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 配置（全部数值取自《实验设置与计算范围.md》）
# ---------------------------------------------------------------------------
@dataclass
class SDFInitConfig:
    # 几何（§3）
    lx: float = 1.6
    ly: float = 0.5
    # Heaviside / 过渡带（§7）
    beta: float = 0.01
    offset: float = 0.0  # 零水平集压在域边界上（边界 S = 0.5）
    # 边界带（《SDF初始化说明》§4；带宽取敏感带 4β ≈ 0.04，§7）
    band_width: float = 0.04
    w_bnd: float = 10.0
    # 阶段 A：Adam（§11、§12）
    adam_steps: int = 12000
    adam_lr: float = 1e-3
    adam_n_interior: int =4096
    adam_n_band: int = 1024
    # 阶段 B：L-BFGS（§11、§12）
    lbfgs_blocks: int = 40
    lbfgs_max_iter: int = 100
    lbfgs_n_interior: int = 32768
    lbfgs_n_band: int = 4096
    # 独立验证集（只评估不训练，规模自定）
    val_n_interior: int = 8192
    val_n_band: int = 2048
    # 早停：连续 patience 块验证损失相对改善 < rtol 则停
    early_stop_patience: int = 3
    early_stop_rtol: float = 1e-4
    # 日志 / 绘图
    print_every: int = 100
    plot_every: int = 500
    # 输出
    fig_dir: str = os.path.join(_DIR, "sdf_figures")
    weights_dir: str = os.path.join(_DIR, "weights")
    seed: int = 20260902
    dtype: torch.dtype = torch.float64


# ---------------------------------------------------------------------------
# 目标场与采样
# ---------------------------------------------------------------------------
def phi0_target(xy: torch.Tensor, cfg: SDFInitConfig) -> torch.Tensor:
    """全材料距离函数。xy: (N, 2) -> (N, 1)（逐点对应，不要再额外 unsqueeze）。"""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    d = torch.minimum(torch.minimum(x, cfg.lx - x), torch.minimum(y, cfg.ly - y))
    return d + cfg.offset


def sample_interior_uniform(
    n: int, cfg: SDFInitConfig, g: torch.Generator
) -> torch.Tensor:
    """域内均匀采样 (n, 2)。"""
    xy = torch.rand(n, 2, generator=g, dtype=cfg.dtype)
    return xy * torch.tensor([cfg.lx, cfg.ly], dtype=cfg.dtype)


def sample_interior_sobol(n: int, cfg: SDFInitConfig, seed: int) -> torch.Tensor:
    """域内 Sobol 低差异采样 (n, 2)（用于 L-BFGS 固定点集 / 验证集）。"""
    eng = SobolEngine(dimension=2, scramble=True, seed=seed)
    pts = eng.draw(n, dtype=cfg.dtype)
    return pts * torch.tensor([cfg.lx, cfg.ly], dtype=cfg.dtype)


def sample_boundary_band(
    n: int, cfg: SDFInitConfig, g: torch.Generator
) -> torch.Tensor:
    """沿四边内侧窄带采样，返回 (n + 4, 2)（n 个带内点 + 4 个角点）。

    - 按边长比例分配到四条边（最大余数法保证总数恰为 n）；
    - 每边内：沿边位置 s ~ U(0, 边长)，到边距离 d 隔点取 0（严格压在边上），
      其余 d ~ U(0, band_width]；
    - 外加 4 个角点。
    """
    edges = [("bottom", cfg.lx), ("top", cfg.lx), ("left", cfg.ly), ("right", cfg.ly)]
    perim = 2.0 * (cfg.lx + cfg.ly)

    raw = [n * L / perim for _, L in edges]
    counts = [int(r) for r in raw]
    rem = n - sum(counts)
    order = sorted(range(4), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:rem]:
        counts[i] += 1

    pts: List[torch.Tensor] = []
    for (name, L), m in zip(edges, counts):
        if m == 0:
            continue
        s = torch.rand(m, 1, generator=g, dtype=cfg.dtype) * L
        d = torch.rand(m, 1, generator=g, dtype=cfg.dtype) * cfg.band_width
        d[0::2] = 0.0  # 隔点严格压在边上
        if name == "bottom":
            p = torch.cat([s, d], dim=1)
        elif name == "top":
            p = torch.cat([s, cfg.ly - d], dim=1)
        elif name == "left":
            p = torch.cat([d, s], dim=1)
        else:  # right
            p = torch.cat([cfg.lx - d, s], dim=1)
        pts.append(p)

    corners = torch.tensor(
        [[0.0, 0.0], [cfg.lx, 0.0], [0.0, cfg.ly], [cfg.lx, cfg.ly]],
        dtype=cfg.dtype,
    )
    return torch.cat(pts + [corners], dim=0)


# ---------------------------------------------------------------------------
# 前向与损失
# ---------------------------------------------------------------------------
def eval_phi(phi: torch.nn.Module, xy: torch.Tensor) -> torch.Tensor:
    """在 t = 0 切片上评估 phi。xy: (N, 2) -> (N, 1)。"""
    t = torch.zeros(xy.shape[0], 1, dtype=xy.dtype, device=xy.device)
    return phi(torch.cat([xy, t], dim=1))


def sdf_loss(
    phi: torch.nn.Module,
    xy_interior: torch.Tensor,
    xy_band: torch.Tensor,
    cfg: SDFInitConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (总损失, 内部 MSE, 边界带 MSE)。预测与目标均为 (N, 1)，逐点对应。"""
    pred_i = eval_phi(phi, xy_interior)
    pred_b = eval_phi(phi, xy_band)
    loss_i = torch.mean((pred_i - phi0_target(xy_interior, cfg)) ** 2)
    loss_b = torch.mean((pred_b - phi0_target(xy_band, cfg)) ** 2)
    return loss_i + cfg.w_bnd * loss_b, loss_i, loss_b


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_snapshot(phi: torch.nn.Module, cfg: SDFInitConfig, title: str, path: str) -> None:
    """保存当前 phi 的四联图：预测 phi、目标 phi0、绝对误差、材料场 S。"""
    nx, ny = 321, 101
    xs = torch.linspace(0.0, cfg.lx, nx, dtype=cfg.dtype)
    ys = torch.linspace(0.0, cfg.ly, ny, dtype=cfg.dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")  # (ny, nx)
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    with torch.no_grad():
        pred = eval_phi(phi, grid).reshape(ny, nx)
    tgt = phi0_target(grid, cfg).reshape(ny, nx)
    err = (pred - tgt).abs()
    S = 0.5 * (1.0 + torch.tanh(pred / (2.0 * cfg.beta)))

    Xn, Yn = X.numpy(), Y.numpy()
    panels = [
        (pred.numpy(), "phi (pred)", "viridis"),
        (tgt.numpy(), "phi0 (target)", "viridis"),
        (err.numpy(), "|phi - phi0|", "hot"),
        (S.numpy(), "S = H(phi)", "viridis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 5.0), constrained_layout=True)
    for ax, (Z, name, cmap) in zip(axes.ravel(), panels):
        pc = ax.pcolormesh(Xn, Yn, Z, cmap=cmap, shading="auto")
        ax.set_aspect("equal")
        ax.set_title(name)
        fig.colorbar(pc, ax=ax, shrink=0.85)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(
    adam_hist: List[Tuple[int, float, float, float]],
    lbfgs_hist: List[Dict],
    cfg: SDFInitConfig,
    path: str,
) -> None:
    """训练结束后保存损失曲线（Adam 按打印点、L-BFGS 按块）。"""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if adam_hist:
        steps = [h[0] for h in adam_hist]
        ax.semilogy(steps, [h[1] for h in adam_hist], label="Adam total")
        ax.semilogy(steps, [h[2] for h in adam_hist], "--", label="Adam interior")
        ax.semilogy(steps, [h[3] for h in adam_hist], "--", label="Adam band")
    if lbfgs_hist:
        offset = adam_hist[-1][0] if adam_hist else 0
        xs = [offset + h["block"] * cfg.lbfgs_max_iter for h in lbfgs_hist]
        ax.semilogy(xs, [h["train"] for h in lbfgs_hist], "o-", label="L-BFGS train")
        ax.semilogy(xs, [h["val"] for h in lbfgs_hist], "s-", label="L-BFGS val")
    ax.set_xlabel("step (L-BFGS 段横轴 = Adam 步数 + 块号 x 70)")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 阶段 A：Adam（每步重采样）
# ---------------------------------------------------------------------------
def train_adam(
    phi: torch.nn.Module, cfg: SDFInitConfig, g: torch.Generator
) -> List[Tuple[int, float, float, float]]:
    print(f"[阶段 A] Adam：{cfg.adam_steps} 步，lr={cfg.adam_lr}，"
          f"每步重采样 {cfg.adam_n_interior} 内部点 + {cfg.adam_n_band} 边界带点")
    opt = torch.optim.Adam(phi.parameters(), lr=cfg.adam_lr)
    hist: List[Tuple[int, float, float, float]] = []
    t0 = time.time()
    for step in range(1, cfg.adam_steps + 1):
        xy_i = sample_interior_uniform(cfg.adam_n_interior, cfg, g)
        xy_b = sample_boundary_band(cfg.adam_n_band, cfg, g)
        loss, loss_i, loss_b = sdf_loss(phi, xy_i, xy_b, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.print_every == 0 or step == 1:
            print(f"  Adam step {step:5d}/{cfg.adam_steps} | "
                  f"total={loss.item():.6e} | interior={loss_i.item():.6e} | "
                  f"band={loss_b.item():.6e} | elapsed={time.time() - t0:.1f}s")
            hist.append((step, loss.item(), loss_i.item(), loss_b.item()))
        if step % cfg.plot_every == 0:
            plot_snapshot(
                phi, cfg, f"Adam step {step}",
                os.path.join(cfg.fig_dir, f"adam_step{step:05d}.png"),
            )
    return hist


# ---------------------------------------------------------------------------
# 阶段 B：L-BFGS 精调（固定点集 + 独立验证集 + 早停 + 回滚）
# ---------------------------------------------------------------------------
def train_lbfgs(
    phi: torch.nn.Module, cfg: SDFInitConfig, g: torch.Generator
) -> List[Dict]:
    print(f"[阶段 B] L-BFGS：{cfg.lbfgs_blocks} 块 x {cfg.lbfgs_max_iter} 次，"
          f"固定 {cfg.lbfgs_n_interior} Sobol 内部点 + {cfg.lbfgs_n_band} 边界带点，"
          f"验证集 {cfg.val_n_interior} + {cfg.val_n_band}（只评估）")

    # 固定训练点集（Sobol 内部点 + 边界带点）与独立验证集（不同种子，只评估不训练）
    xy_i = sample_interior_sobol(cfg.lbfgs_n_interior, cfg, seed=cfg.seed + 1)
    xy_b = sample_boundary_band(cfg.lbfgs_n_band, cfg, g)
    xy_vi = sample_interior_sobol(cfg.val_n_interior, cfg, seed=cfg.seed + 2)
    xy_vb = sample_boundary_band(cfg.val_n_band, cfg, g)

    hist: List[Dict] = []
    best_val = float("inf")
    best_state = copy.deepcopy(phi.state_dict())
    prev_val = float("inf")
    stall = 0

    for block in range(1, cfg.lbfgs_blocks + 1):
        opt = torch.optim.LBFGS(
            phi.parameters(),
            max_iter=cfg.lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-16,
            tolerance_change=1e-16,
        )

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss, _, _ = sdf_loss(phi, xy_i, xy_b, cfg)
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            train_loss, tr_i, tr_b = sdf_loss(phi, xy_i, xy_b, cfg)
            val_loss, va_i, va_b = sdf_loss(phi, xy_vi, xy_vb, cfg)
        train_v, val_v = train_loss.item(), val_loss.item()
        hist.append({"block": block, "train": train_v, "val": val_v})

        improved = ""
        if val_v < best_val:
            best_val = val_v
            best_state = copy.deepcopy(phi.state_dict())
            improved = " *best"
        rel_impr = (prev_val - val_v) / max(abs(prev_val), 1e-300)
        stall = stall + 1 if rel_impr < cfg.early_stop_rtol else 0
        prev_val = val_v

        print(f"  L-BFGS block {block:3d}/{cfg.lbfgs_blocks} | "
              f"train={train_v:.6e} (i={tr_i.item():.3e}, b={tr_b.item():.3e}) | "
              f"val={val_v:.6e} (i={va_i.item():.3e}, b={va_b.item():.3e}) | "
              f"rel_impr={rel_impr:.3e}{improved}")
        plot_snapshot(
            phi, cfg, f"L-BFGS block {block} (val={val_v:.3e})",
            os.path.join(cfg.fig_dir, f"lbfgs_block{block:03d}.png"),
        )

        if stall >= cfg.early_stop_patience:
            print(f"  早停：连续 {stall} 块验证损失相对改善 < {cfg.early_stop_rtol}")
            break

    # 回滚到验证损失最优的权重（防止过拟合固定配点）
    phi.load_state_dict(best_state)
    print(f"  回滚到验证损失最优权重：best val = {best_val:.6e}")
    return hist


# ---------------------------------------------------------------------------
# 诊断（训练后自检，对应《SDF初始化说明》§7）
# ---------------------------------------------------------------------------
def diagnose(phi: torch.nn.Module, cfg: SDFInitConfig) -> None:
    print("[诊断]")
    nx, ny = 641, 201
    xs = torch.linspace(0.0, cfg.lx, nx, dtype=cfg.dtype)
    ys = torch.linspace(0.0, cfg.ly, ny, dtype=cfg.dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # 拟合误差：全域 max、边界带内 max
    with torch.no_grad():
        pred = eval_phi(phi, grid)
        tgt = phi0_target(grid, cfg)
    err = (pred - tgt).abs()
    band_mask = (tgt - cfg.offset).squeeze(1) <= cfg.band_width
    print(f"  拟合误差：全域 max = {err.max().item():.6e} | "
          f"边界带(|phi0|<={cfg.band_width}) max = {err[band_mask].max().item():.6e} "
          f"(2beta = {2.0 * cfg.beta:.3e}，带内误差应明显小于它)")

    # |grad phi| 统计：距离函数应处处 ≈ 1（中轴线折点附近被网络抹平、偏小属正常）
    grads: List[torch.Tensor] = []
    chunk = 32768
    for s in range(0, grid.shape[0], chunk):
        xy_g = grid[s:s + chunk].clone().requires_grad_(True)
        p = eval_phi(phi, xy_g)
        (gp,) = torch.autograd.grad(p.sum(), xy_g)
        grads.append(gp.detach())
    gn = torch.cat(grads, dim=0).norm(dim=1)
    print(f"  |grad phi|：mean = {gn.mean().item():.6f} | min = {gn.min().item():.6f} | "
          f"max = {gn.max().item():.6f}（距离函数应 ≈ 1）")

    # 边界上的 S 值（offset = 0 时应为 0.5）
    n_edge = 2000
    on_edge = torch.cat([
        torch.stack([torch.linspace(0.0, cfg.lx, n_edge, dtype=cfg.dtype),
                     torch.zeros(n_edge, dtype=cfg.dtype)], dim=1),
        torch.stack([torch.linspace(0.0, cfg.lx, n_edge, dtype=cfg.dtype),
                     torch.full((n_edge,), cfg.ly, dtype=cfg.dtype)], dim=1),
        torch.stack([torch.zeros(n_edge, dtype=cfg.dtype),
                     torch.linspace(0.0, cfg.ly, n_edge, dtype=cfg.dtype)], dim=1),
        torch.stack([torch.full((n_edge,), cfg.lx, dtype=cfg.dtype),
                     torch.linspace(0.0, cfg.ly, n_edge, dtype=cfg.dtype)], dim=1),
    ], dim=0)
    with torch.no_grad():
        S_edge = 0.5 * (1.0 + torch.tanh(eval_phi(phi, on_edge) / (2.0 * cfg.beta)))
    print(f"  边界上的 S：mean = {S_edge.mean().item():.6f} | "
          f"min = {S_edge.min().item():.6f} | max = {S_edge.max().item():.6f}"
          f"（offset={cfg.offset}，应为 0.5）")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(cfg: Optional[SDFInitConfig] = None) -> None:
    cfg = cfg or SDFInitConfig()
    os.makedirs(cfg.fig_dir, exist_ok=True)
    os.makedirs(cfg.weights_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)

    print("=" * 72)
    print("SDF 初始化：phi(x, y, t=0) 拟合 phi0 = min(x, Lx-x, y, Ly-y)")
    print(f"  域 [0,{cfg.lx}]x[0,{cfg.ly}] | beta={cfg.beta} | offset={cfg.offset} | "
          f"band_width={cfg.band_width} | w_bnd={cfg.w_bnd} | dtype={cfg.dtype}")
    print("=" * 72)

    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype)
    phi = build_sdf_network(net_cfg)

    # 初始状态图
    plot_snapshot(phi, cfg, "initial (untrained)",
                  os.path.join(cfg.fig_dir, "step000000_initial.png"))

    adam_hist = train_adam(phi, cfg, g)
    lbfgs_hist = train_lbfgs(phi, cfg, g)

    plot_history(adam_hist, lbfgs_hist, cfg,
                 os.path.join(cfg.fig_dir, "loss_history.png"))
    plot_snapshot(phi, cfg, "final (after rollback)",
                  os.path.join(cfg.fig_dir, "final.png"))

    # 保存权重（文件名与《网络初始化说明》约定一致：phi_init.pt）
    save_path = os.path.join(cfg.weights_dir, "phi_init.pt")
    torch.save(phi.state_dict(), save_path)
    print(f"[保存] SDF 网络权重 -> {save_path}")

    diagnose(phi, cfg)
    print("[完成] 图片输出目录：" + cfg.fig_dir)


if __name__ == "__main__":
    main()
