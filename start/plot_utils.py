# -*- coding: utf-8 -*-
"""plot_utils.py — 统一出图风格（2026-09-24 起）。

约定：
  1. 一图一文件：每个 PNG 只含一个面板；循环产物按类型分文件夹保存
     （fig_path 的 kind 即子文件夹名，tag 即文件名，如 vn/iter001.png）；
  2. 颜色以黑白为主：材料/能量类正量用 gray_r（黑 = 大/有材料），有符号场
     用 gray 对称色标（黑 = 负、白 = 正、零 = 中灰），曲线图黑/灰 + 线型区分；
  3. 中间硬块（重块，不可优化）一律用红框标记；四角冻结区不标记。
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # 无界面后端，只保存图片
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# 色图约定
CMAP_MAT = "gray_r"   # 材料场 S / 正量密度：黑 = 1/大，白 = 0/小
CMAP_SEQ = "gray_r"   # 正量场（能量密度、|u| 等）：黑 = 大
CMAP_DIV = "gray"     # 有符号场（V_n、φ、ΔS、应力等）：黑 = 负，白 = 正

BLOCK_COLOR = "red"   # 硬块标记颜色（图中唯一的彩色元素）

# 曲线图黑白线型循环（黑实、灰虚、灰点划、黑点线）
LINE_STYLES = [
    dict(color="k", ls="-"),
    dict(color="0.35", ls="--"),
    dict(color="0.55", ls="-."),
    dict(color="k", ls=":"),
    dict(color="0.7", ls="-"),
]


def setup_style():
    """统一字体与负号（Windows 自带微软雅黑/黑体）。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def fig_path(fig_dir: str, kind: str, tag: str = "") -> str:
    """拼出图路径：tag 非空 → {fig_dir}/{kind}/{tag}.png；否则 {fig_dir}/{kind}.png。

    kind 即类型文件夹名（vn / geometry / loss_hjb_adam / history …），
    tag 在交替循环中取 iter{k:03d}，单独运行时为空（平铺文件名）。
    """
    if tag:
        return os.path.join(fig_dir, kind, f"{tag}.png")
    return os.path.join(fig_dir, f"{kind}.png")


def add_block_box(ax, block_x: Tuple[float, float], block_y: Tuple[float, float]):
    """中间硬块红框标记（四角冻结区按约定不标记）。"""
    ax.add_patch(Rectangle(
        (block_x[0], block_y[0]), block_x[1] - block_x[0], block_y[1] - block_y[0],
        fill=False, ec=BLOCK_COLOR, lw=1.5, zorder=5))


def save_field(path: str, X, Y, Z, *, lx: float, ly: float, title: str = "",
               cmap: str = CMAP_SEQ, clim: Optional[Tuple[float, float]] = None,
               symmetric: bool = False,
               block: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
               contours: Optional[Sequence[Tuple[object, str, str, str]]] = None,
               dpi: int = 150):
    """单面板场图（pcolormesh，等比例，colorbar）。

    X/Y/Z 为 (ny, nx) numpy 数组；symmetric=True 时色标取 ±max|Z| 对称
    （有符号场，零 = 中灰）；block = (block_x, block_y) 画红框；
    contours = [(Zc, color, ls, label)] 画零等值线（label 非空时建图例）。
    """
    import numpy as np

    Z = np.asarray(Z, dtype=float)
    if symmetric:
        lim = float(np.nanmax(np.abs(Z))) if Z.size else 1.0
        clim = (-max(lim, 1e-30), max(lim, 1e-30))
    kw = dict(vmin=clim[0], vmax=clim[1]) if clim is not None else {}

    fig, ax = plt.subplots(figsize=(7.6, 7.6 * ly / lx + 1.0),
                           constrained_layout=True)
    pc = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", **kw)
    fig.colorbar(pc, ax=ax, shrink=0.85)
    if contours:
        handles = []
        for Zc, color, ls, label in contours:
            ax.contour(X, Y, Zc, levels=[0.0], colors=[color], linestyles=[ls],
                       linewidths=1.4)
            if label:
                handles.append(Line2D([0], [0], color=color, ls=ls, lw=1.4,
                                      label=label))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8)
    if block is not None:
        add_block_box(ax, block[0], block[1])
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=10)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_lines(path: str, series: Sequence[Tuple[object, object, str]], *,
               title: str = "", xlabel: str = "", ylabel: str = "",
               logy: bool = False,
               hlines: Optional[Sequence[Tuple[float, str]]] = None,
               figsize: Tuple[float, float] = (7.2, 4.2), dpi: int = 130):
    """单面板曲线图（黑白线型自动循环）。

    series = [(x, y, label)]；hlines = [(y, label)] 画水平参考线（灰虚线）。
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plot = ax.semilogy if logy else ax.plot
    for i, (x, y, label) in enumerate(series):
        st = LINE_STYLES[i % len(LINE_STYLES)]
        plot(x, y, label=label, lw=1.3,
             marker=["o", "s", "^", "D", "v"][i % 5], markersize=3.5, **st)
    if hlines:
        for y0, label in hlines:
            ax.axhline(y0, color="0.4", ls="--", lw=0.9, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    if any(lbl for _, _, lbl in series) or hlines:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
