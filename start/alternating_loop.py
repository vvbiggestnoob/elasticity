# -*- coding: utf-8 -*-
"""
alternating_loop.py — 拓扑优化外层交替循环（Step A 力学重解 ↔ Step B HJB 演化）

在 sdf_init.py（初始 SDF）与 mech_init.py（初始力学）的产物之上，交替执行：
    第 k 次迭代（k = 1..n_iter，t_k = k·dt）：
      Step B  HJB 演化：冻结力学 mech_{k-1}，phi 从 phi_{k-1} 热启动，
              在时间窗口 [t_{k-1}, t_k] 上解 HJB，输出 phi_k；
      Step A  力学重解：在新几何 phi_k 的 t = t_k 切片上，从 mech_{k-1} 热启动
              重训 5 个力学网络，输出 mech_k，供下一次迭代使用。

变量传递约定（本脚本的核心，改动前务必理解）：
  1. 伪时间线程：phi_k 是窗口 [t_{k-1}, t_k] 上训练出的时空网络；"当前几何"
     永远是其右端切片 t = t_k。HJB 步内 phi_ref 在 t_current = t_{k-1} 取值
     （hjb_step_eik 内部处理）；力学重解通过 MechInitConfig.phi_t = t_k 在同一
     切片取材料场（heaviside_s）。两处切片必须一致，错一切片即错位一个 dt。
  2. 权重命名链：SDF 为 phi_init.pt → phi_iter1.pt → phi_iter2.pt …；
     力学为 *_init.pt → *_iter1.pt → *_iter2.pt …（u_x/u_y/sigma_xx/sigma_xy/
     sigma_yy 五个）。每步写新文件，不覆盖上游，任何一步可单独复查。
  3. ω²_FEM 锚点传递：mech_init 的平衡方程残差与 Rayleigh 锚定都使用配置里的
     固定 ω²_FEM，几何变了锚点必须更新。HJB 步诊断（fem_check=True）会在演化后
     几何上做 FEM 对拍并写入 hjb_state_iter{k}.json 的 omega2_fem_after，
     本脚本读取该值作为本次力学重解的锚点；FEM 缺失时依次回退
     PINN Rayleigh 商 → 上一步锚点，并打印警告。
  4. 热启动链：phi 的热启动由 hjb_step_eik 内部完成（phi 与 phi_ref 同权重
     加载）；力学的热启动由 MechInitConfig.init_suffix 指向上一迭代的权重。
  5. 种子：HJB 步 seed = seed + k，力学重解 seed = seed + 100000 + k，
     每次迭代点集不同但全程确定可复现。
  6. 罚参数自适应（Step D，alpha_adaptive=True 时）：自第 1 步起按
     1/α = min(τ·ω²/(ΔV)², α_max) 更新（《HJB方程推导与特征值拓扑优化》
     §3.5 / 《方案_详细版》§5.5；τ = 1.1，α_max = 1e6）。第 1 步用初始
     几何面积（initial_area）与初始 ω² 锚点；之后每步 HJB 结束后用当步
     FEM 对拍锚点与 area_after 算下一步的 α。四角已由 corner_freeze 硬
     非设计域冻结，原 p_cap 角部保护方案于 2026-09-18 移除。

图片策略（quiet_figures=True）：子模块每步只存 vn_field / final / loss_history
三张图到 loop_figures/iter{k:03d}_hjb|mech/；循环级只维护两张汇总图
（history.png 收敛曲线、zero_level_sets.png 零水平集叠加），每次迭代重绘覆盖。

加速机制（利用小 dt 下每步变化小的特性）：
  1. 预测-校正分裂：每 full_step_every 步跑一次完整四损失校正步，其余为廉价
     预测步（lam_r=lam_eik=0，L_0+L_anchor 纯回归，无自动微分二阶图，每步成本
     低数倍）；廉价步后诊断仍评估真实 HJB 残差，超过 cheap_residual_tol 时
     下一步强制完整校正步（护栏）。
  2. 力学跳步：max|Δφ|（phi_k 在窗口两端切片的网格最大差）< mech_skip_tol 时
     跳过力学重解，把 mech_{k-1} 权重复制为 mech_k（命名链保持连续）；
     ω² 锚点仍由每步 FEM 对拍照常更新，下次真正重解时使用最新值。
  3. 停滞检测与早停：Adam 停滞提前切 L-BFGS、L-BFGS 验证早停，循环中默认
     开启（hjb_extra / mech_extra 可覆盖）。

HJB 单步模块默认为 hjb_step_eik（含 eikonal 正则）；想换回无 eikonal 的
hjb_step.py，把下方 import 改一行即可（lam_eik / quiet_figures 会被自动忽略，
但旧版没有精简存图，过程图会很多）。

全程 float64。所有相对路径相对本脚本所在目录解析。
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 无界面后端，只保存图片
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import torch

# 所有相对路径相对本脚本所在目录解析，从任意工作目录运行均可
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import mech_init
import hjb_step_eik as hjb_mod  # 换回原始版本：import hjb_step as hjb_mod
from networks import NetworkConfig, build_sdf_network, MECH_NET_NAMES


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class LoopConfig:
    # 几何（与两个子模块默认一致，显式传递避免漂移）
    lx: float = 1.6
    ly: float = 0.5

    # 循环规模与演化参数
    n_iter: int = 1000               # 交替迭代次数，总伪时间 T = n_iter·dt
    dt: float = 0.005               # 每步伪时间步长（CFL：|V_n|·dt ≲ β=0.01 → |V_n| ≲ 2；
                                    #  罚压力峰值 ≈ √(τ·ω²·α_max)，超限需调小 dt 或 α_max）
    alpha: float = 1.0             # 固定罚参数（仅 alpha_adaptive=False 时生效；
                                   #  自适应开启时第 1 步起即用公式，见 main 前段）
    v_target: float = 0.4          # C = 50% × 0.8

    # Step D：罚参数自适应（《HJB方程推导与特征值拓扑优化》§3.5 /
    #   《方案_详细版》§5.5）：1/α = min(τ·ω²/(ΔV)², α_max)，ΔV = ∫S dx − C。
    #   四角已由 corner_freeze 硬冻结，原 p_cap 角部保护方案于 2026-09-18 移除。
    #   注意：罚压力 p = min(τ·ω²/|ΔV|, α_max·|ΔV|) 在 |ΔV| ≈ √(τ·ω²/α_max)
    #   处达峰值 √(τ·ω²·α_max)——τ=1.1、ω²≈0.225、α_max=1e6 时峰值 ≈ 500，
    #   远超 dt=0.005 的 CFL 设计（|V_n| ≲ 2）；后期若过冲需调小 dt 或 α_max
    #   （α_max ≲ 16 可把峰值压回 2 以内）。
    alpha_adaptive: bool = True
    tau: float = 1.1               # 文档 §3.5 的缩放因子
    alpha_max: float = 1e6         # 罚系数 1/α 上限（《方案_详细版》§5.5）

    # HJB 损失权重：None = 沿用 hjb 模块默认值（避免两处定义漂移）
    lam_0: Optional[float] = None
    lam_anchor: Optional[float] = None
    lam_eik: Optional[float] = None

    # 角部硬非设计域（透传给 hjb 模块；0 = 关闭）。冻结四角 corner_freeze 见方
    # 区域为材料，强制保留角部（桥式结构的拱脚）。2026-09-14 起，用于解决
    # 角部 inflow 边界 + 网络平滑导致的角部误侵蚀（见 exp01_bridge/README.md）。
    corner_freeze: float = 0.0
    w_corner_freeze: float = 10.0

    # 每个完整 HJB 步的训练量（热启动 + 停滞检测/早停，较单步独立运行已压缩）
    hjb_adam_steps: int = 1500
    hjb_lbfgs_blocks: int = 100

    # 力学重解训练量（热启动，几何每步只走 dt，远小于初始化的 20000/160）；
    # Adam 固定步数（循环中不做停滞检测），L-BFGS 块数只是安全上限，
    # 实际由早停规则决定（train 逐块相对下降连续 5 块 < early_stop_rtol 即退出）
    mech_adam_steps: int = 1500
    mech_lbfgs_blocks: int = 100

    # L-BFGS 早停阈值：HJB 与力学重解共用，train 损失相对改善连续 5 块
    # 低于该值即退出（patience=5 为两模块各自默认；hjb_extra/mech_extra 可覆盖）
    early_stop_rtol: float = 0.01

    # 预测-校正分裂：每 full_step_every 步一个完整校正步，其余为廉价预测步
    # （lam_r=lam_eik=0 纯回归；1 = 关闭分裂，每步完整）
    full_step_every: int = 1
    cheap_adam_steps: int = 1000
    cheap_lbfgs_blocks: int = 20
    # 廉价步护栏：HJB 残差 RMS(验证集) 超过该值时下一步强制完整校正步；
    # None = 关闭护栏（残差量级随配置变，先观察 hjb_state 的 residual_rms_val 再设）
    cheap_residual_tol: Optional[float] = None

    # 力学跳步：max|Δφ|（新几何窗口两端切片的网格最大差）< mech_skip_tol 时
    # 跳过重解、沿用上一接力学（0 = 不跳过；阈值参考界面厚度 β=0.01）
    mech_skip_tol: float = 0

    # 每个新几何的 FEM ω² 对拍：同时是力学重解锚点的来源；关闭则回退 PINN Rayleigh
    fem_check: bool = True

    # 额外覆盖（逃生舱）：透传给子模块配置，自动过滤不识别的字段
    hjb_extra: Optional[dict] = None
    mech_extra: Optional[dict] = None

    # 路径
    weights_dir: str = "weights"
    phi_init_name: str = "phi_init.pt"      # sdf_init.py 的产物
    mech_init_suffix: str = "_init.pt"      # mech_init.py 的产物
    loop_fig_dir: str = "loop_figures"
    history_out: str = "weights/loop_history.json"
    seed: int = 20260909

    dtype: torch.dtype = torch.float64


# ---------------------------------------------------------------------------
# 命名链（变量传递约定 2）
# ---------------------------------------------------------------------------

def phi_name(k: int, cfg: LoopConfig) -> str:
    """第 k 步几何的 SDF 权重文件名；k=0 为初始 SDF。"""
    return cfg.phi_init_name if k == 0 else f"phi_iter{k}.pt"


def mech_suffix(k: int, cfg: LoopConfig) -> str:
    """第 k 步几何对应力学网络的文件名后缀；k=0 为初始力学。"""
    return cfg.mech_init_suffix if k == 0 else f"_iter{k}.pt"


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """按 dataclass 字段过滤 kwargs，换模块版本时不识别字段自动忽略并提示。"""
    valid = {f.name for f in dataclasses.fields(cls)}
    dropped = [k for k in kwargs if k not in valid]
    if dropped:
        print(f"[提示] {cls.__name__} 不识别参数 {dropped}，已忽略")
    return {k: v for k, v in kwargs.items() if k in valid}


# ---------------------------------------------------------------------------
# Step B：HJB 单次演化（调用 hjb_step_eik.main）
# ---------------------------------------------------------------------------

def run_hjb_step(k: int, cfg: LoopConfig, cheap: bool, alpha: float) -> Dict:
    """第 k 步 HJB 演化：窗口 [t_{k-1}, t_k]，返回 hjb_state_iter{k}.json 的内容。

    cheap=True 为廉价预测步：强制 lam_r=lam_eik=0（L_0+L_anchor 纯回归，
    无自动微分二阶图），训练量取 cheap_*；诊断仍评估真实 HJB 残差供护栏判断。
    alpha 为本步罚参数（Step D 每步更新，见 update_alpha）。
    """
    t0 = (k - 1) * cfg.dt
    state_out = _resolve(os.path.join(cfg.weights_dir, f"hjb_state_iter{k}.json"))
    kwargs = dict(
        lx=cfg.lx, ly=cfg.ly,
        t_current=t0, dt=cfg.dt, alpha=alpha, v_target=cfg.v_target,
        adam_steps=cfg.cheap_adam_steps if cheap else cfg.hjb_adam_steps,
        lbfgs_blocks=cfg.cheap_lbfgs_blocks if cheap else cfg.hjb_lbfgs_blocks,
        phi_path=_resolve(os.path.join(cfg.weights_dir, phi_name(k - 1, cfg))),
        mech_dir=_resolve(cfg.weights_dir),
        mech_suffix=mech_suffix(k - 1, cfg),
        phi_out=_resolve(os.path.join(cfg.weights_dir, phi_name(k, cfg))),
        state_out=state_out,
        fig_dir=_resolve(os.path.join(cfg.loop_fig_dir, f"iter{k:03d}_hjb")),
        fem_check=cfg.fem_check,
        quiet_figures=True,
        # 角部硬非设计域（0 = 关闭）
        corner_freeze=cfg.corner_freeze,
        w_corner_freeze=cfg.w_corner_freeze,
        # 停滞检测 + L-BFGS 早停（循环默认开启；hjb_extra 可覆盖）
        adam_early_switch=True,
        adam_switch_min_step=200 if cheap else 600,
        adam_switch_window=100 if cheap else 300,
        lbfgs_early_stop=True,
        early_stop_metric="train",   # 早停看训练损失（拟合到收敛才停）
        early_stop_rtol=cfg.early_stop_rtol,   # 阈值与力学重解共用
        seed=cfg.seed + k,
    )
    # 损失权重：None 时沿用模块默认
    for w in ("lam_0", "lam_anchor", "lam_eik"):
        v = getattr(cfg, w)
        if v is not None:
            kwargs[w] = v
    if cheap:
        # 廉价预测步：关闭 PDE 残差与 eikonal（在权重覆盖之后，强制生效）
        kwargs["lam_r"] = 0.0
        kwargs["lam_eik"] = 0.0
    if cfg.hjb_extra:
        kwargs.update(cfg.hjb_extra)
    hcfg = hjb_mod.HJBConfig(**_filter_kwargs(hjb_mod.HJBConfig, kwargs))
    hjb_mod.main(hcfg)
    with open(state_out, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step A：力学重解（调用 mech_init.main，热启动 + 新几何时间切片 + 新 ω² 锚点）
# ---------------------------------------------------------------------------

def run_mech_solve(k: int, cfg: LoopConfig, omega2_fem: float):
    """第 k 步力学重解：几何 = phi_k 在 t = t_k 的切片，返回诊断字典。"""
    kwargs = dict(
        lx=cfg.lx, ly=cfg.ly,
        phi_path=_resolve(os.path.join(cfg.weights_dir, phi_name(k, cfg))),
        phi_t=k * cfg.dt,                     # 变量传递约定 1：右端切片
        init_suffix=mech_suffix(k - 1, cfg),  # 变量传递约定 4：热启动
        save_suffix=mech_suffix(k, cfg),
        omega2_fem=omega2_fem,                # 变量传递约定 3：新几何锚点
        adam_steps=cfg.mech_adam_steps,
        lbfgs_blocks=cfg.mech_lbfgs_blocks,
        weights_dir=_resolve(cfg.weights_dir),
        fig_dir=_resolve(os.path.join(cfg.loop_fig_dir, f"iter{k:03d}_mech")),
        quiet_figures=True,
        # Adam 固定步数（不做停滞检测）；L-BFGS 早停看 train 逐块相对下降，
        # 连续 5 块 < early_stop_rtol 即退出，块数只作安全上限（mech_extra 可覆盖）
        adam_early_switch=False,
        lbfgs_early_stop=True,
        early_stop_metric="train",
        early_stop_rtol=cfg.early_stop_rtol,
        seed=cfg.seed + 100000 + k,
    )
    if cfg.mech_extra:
        kwargs.update(cfg.mech_extra)
    mcfg = mech_init.MechInitConfig(**_filter_kwargs(mech_init.MechInitConfig, kwargs))
    return mech_init.main(mcfg)


def omega2_anchor_from_state(state: Dict, prev: float) -> float:
    """变量传递约定 3：优先 FEM 对拍值，依次回退 PINN Rayleigh、上一步锚点。"""
    w = state.get("omega2_fem_after")
    if w is not None:
        return float(w)
    w = state.get("omega2_after_rayleigh_frozen_mech")
    if w is not None:
        print("[警告] 本步无 FEM ω²（fem_check 关闭或失败），"
              "力学重解锚点回退为 PINN Rayleigh 商（冻结力学近似）")
        return float(w)
    print("[警告] 无法取得新几何 ω²，力学重解锚点沿用上一步值 "
          f"{prev:.6f}")
    return prev


def update_alpha(area: float, omega2: float, cfg: LoopConfig) -> Tuple[float, float]:
    """Step D：罚参数自适应更新（《HJB方程推导与特征值拓扑优化》§3.5 /
    《方案_详细版》§5.5）。

        1/α = min(τ·ω²/(ΔV)², α_max)，ΔV = area − C

    参数：τ = 1.1，α_max = 1e6。返回 (alpha, p)，p = |ΔV|/α 即本状态对应的
    罚压力绝对值。
    """
    dV = abs(area - cfg.v_target)
    inv_alpha = min(cfg.tau * omega2 / max(dV, 1e-12) ** 2, cfg.alpha_max)
    return 1.0 / inv_alpha, dV * inv_alpha


@torch.no_grad()
def initial_area(cfg: LoopConfig) -> float:
    """初始几何（phi_init @ t=0）的 ∫S dx：均匀网格上 S = ½(1+tanh(φ/2β))
    的均值乘域面积（与 rayleigh_and_area 的口径一致，均匀网格更准）。

    用于第 1 步的罚参数计算（与 towangbt 一致：公式自第 1 步起生效）。
    """
    beta = hjb_mod.HJBConfig().beta
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype)
    phi = build_sdf_network(net_cfg)
    phi.load_state_dict(torch.load(
        _resolve(os.path.join(cfg.weights_dir, cfg.phi_init_name)), weights_only=True))
    phi.eval().requires_grad_(False)
    xs = torch.linspace(0.0, cfg.lx, 1601, dtype=cfg.dtype)
    ys = torch.linspace(0.0, cfg.ly, 501, dtype=cfg.dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    tt = torch.zeros(grid.shape[0], 1, dtype=cfg.dtype)
    s_sum = 0.0
    for s0 in range(0, grid.shape[0], 65536):   # ½(1+tanh(φ/2β)) = sigmoid(φ/β)
        v = phi(torch.cat([grid[s0:s0 + 65536], tt[s0:s0 + 65536]], dim=1))
        s_sum += torch.sigmoid(v.squeeze(1) / beta).sum()
    return float(cfg.lx * cfg.ly * s_sum / grid.shape[0])


@torch.no_grad()
def max_dphi(k: int, cfg: LoopConfig) -> float:
    """几何变化量度：phi_k 在窗口两端切片（t_{k-1}, t_k）上的最大逐点差。

    用于力学跳步判定（加速机制 2）。网格取 161×51（与 mech 诊断网格同量级，
    比存图网格粗，足够量度变化）。
    """
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype)
    phi = build_sdf_network(net_cfg)
    phi.load_state_dict(torch.load(
        _resolve(os.path.join(cfg.weights_dir, phi_name(k, cfg))), weights_only=True))
    xs = torch.linspace(0.0, cfg.lx, 161, dtype=cfg.dtype)
    ys = torch.linspace(0.0, cfg.ly, 51, dtype=cfg.dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    tt = torch.full((grid.shape[0], 1), (k - 1) * cfg.dt, dtype=cfg.dtype)
    p0 = phi(torch.cat([grid, tt], dim=1))
    tt.fill_(k * cfg.dt)
    p1 = phi(torch.cat([grid, tt], dim=1))
    return float((p1 - p0).abs().max())


# ---------------------------------------------------------------------------
# 循环级汇总图（每次迭代重绘覆盖，只此两张）
# ---------------------------------------------------------------------------

def save_history_figure(history: List[Dict], cfg: LoopConfig):
    """四联收敛曲线：F、ω²、面积、罚压力 p vs 迭代步。"""
    iters = [h["iter"] for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), constrained_layout=True)

    axes[0].plot(iters, [h["F_before"] for h in history], "o--", color="gray",
                 label="F before (step start)")
    axes[0].plot(iters, [h["F_after"] for h in history], "o-", color="tab:blue",
                 label="F after (HJB step)")
    axes[0].set_title("objective F = −ω² + (1/2α)(∫S−C)²")
    axes[0].legend(fontsize=8)

    om_fem = [h["omega2_fem_after"] for h in history]
    xs_fem = [i for i, v in zip(iters, om_fem) if v is not None]
    ys_fem = [v for v in om_fem if v is not None]
    if xs_fem:
        axes[1].plot(xs_fem, ys_fem, "s-", color="tab:red", label="ω² FEM")
    axes[1].plot(iters, [h["omega2_rayleigh_after"] for h in history], "o--",
                 color="tab:orange", label="ω² Rayleigh (frozen mech)")
    axes[1].set_title("ω² on evolved geometry")
    axes[1].legend(fontsize=8)

    axes[2].plot(iters, [h["area_after"] for h in history], "o-", color="tab:green")
    axes[2].axhline(cfg.v_target, color="k", ls="--", lw=1, label=f"C = {cfg.v_target}")
    axes[2].set_title("area ∫S dx")
    axes[2].legend(fontsize=8)

    # 罚压力 p = |ΔV|/α 轨迹（文档公式下 p = min(τ·ω²/|ΔV|, α_max·|ΔV|)）
    p_used = [h.get("p_used") for h in history]
    xs_p = [i for i, v in zip(iters, p_used) if v is not None]
    ys_p = [v for v in p_used if v is not None]
    if xs_p:
        axes[3].plot(xs_p, ys_p, "o-", color="tab:purple", label="p = |ΔV|/α")
    axes[3].set_title("penalty pressure p = |ΔV|/α")
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
    path = _resolve(os.path.join(cfg.loop_fig_dir, "history.png"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def save_zero_level_figure(k_max: int, cfg: LoopConfig):
    """零水平集叠加图：几何 0..k_max 的 φ=0 等值线（蓝=初始，红=最新）。"""
    net_cfg = NetworkConfig(lx=cfg.lx, ly=cfg.ly, dtype=cfg.dtype)
    phi = build_sdf_network(net_cfg)
    xs = torch.linspace(0.0, cfg.lx, 321, dtype=cfg.dtype)
    ys = torch.linspace(0.0, cfg.ly, 101, dtype=cfg.dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    tt = torch.empty(grid.shape[0], 1, dtype=cfg.dtype)

    fig, ax = plt.subplots(figsize=(11.5, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm")
    handles = []
    for j in range(0, k_max + 1):
        path = _resolve(os.path.join(cfg.weights_dir, phi_name(j, cfg)))
        phi.load_state_dict(torch.load(path, weights_only=True))
        tt.fill_(j * cfg.dt)  # 变量传递约定 1：几何 j 在其窗口右端切片
        z = phi(torch.cat([grid, tt], dim=1)).reshape(X.shape).numpy()
        color = cmap(j / max(k_max, 1))
        ax.contour(X.numpy(), Y.numpy(), z, levels=[0.0], colors=[color],
                   linewidths=1.3)
        if k_max <= 12 or j in (0, k_max):
            handles.append(Line2D([0], [0], color=color, lw=1.3,
                                  label=f"iter {j} (t={j * cfg.dt:.3g})"))

    # 重块区域（不可优化）与域边界
    mcfg0 = mech_init.MechInitConfig()
    bx, by = mcfg0.block_x, mcfg0.block_y
    ax.add_patch(Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0],
                           fill=False, ec="k", ls="--", lw=1.2))
    ax.set_xlim(0, cfg.lx)
    ax.set_ylim(0, cfg.ly)
    ax.set_aspect("equal")
    ax.set_title(f"zero level set evolution (dt = {cfg.dt}, T = {k_max * cfg.dt:.3g})")
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    out = _resolve(os.path.join(cfg.loop_fig_dir, "zero_level_sets.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def main(cfg: Optional[LoopConfig] = None):
    cfg = cfg or LoopConfig()
    cfg.weights_dir = _resolve(cfg.weights_dir)
    cfg.loop_fig_dir = _resolve(cfg.loop_fig_dir)
    cfg.history_out = _resolve(cfg.history_out)
    os.makedirs(cfg.weights_dir, exist_ok=True)
    os.makedirs(cfg.loop_fig_dir, exist_ok=True)

    # 前置检查：初始 SDF 与初始力学权重必须已存在（sdf_init.py / mech_init.py 的产物）
    need = [cfg.phi_init_name] + [n + cfg.mech_init_suffix for n in MECH_NET_NAMES]
    missing = [p for p in need
               if not os.path.exists(os.path.join(cfg.weights_dir, p))]
    if missing:
        raise FileNotFoundError(
            f"缺少初始权重 {missing}（目录 {cfg.weights_dir}）。"
            "请先运行 sdf_init.py 与 mech_init.py。")

    # 初始几何的 ω² 锚点 = mech_init 默认的 FEM 实测值（全材料+重块）
    omega2_anchor = mech_init.MechInitConfig().omega2_fem
    # 罚参数：与 towangbt 一致，第 1 步起即用公式（以初始几何面积与初始 ω² 锚点
    # 计算）；cfg.alpha 仅在 alpha_adaptive=False 时作为固定值使用
    alpha_cur = cfg.alpha
    if cfg.alpha_adaptive:
        area0 = initial_area(cfg)
        alpha_cur, p0 = update_alpha(area0, omega2_anchor, cfg)

    print("=" * 72)
    print(f"交替循环：n_iter = {cfg.n_iter}，dt = {cfg.dt}，"
          f"总伪时间 T = {cfg.n_iter * cfg.dt}，C = {cfg.v_target}")
    if cfg.alpha_adaptive:
        print(f"罚参数：Step D 自适应，1/α = min(τ·ω²/(ΔV)², α_max)，"
              f"τ = {cfg.tau}，α_max = {cfg.alpha_max:g} | "
              f"第 1 步：ΔV = {area0 - cfg.v_target:+.4f}，ω² = {omega2_anchor:.4f} "
              f"→ α = {alpha_cur:.4g}（罚压力 p = {p0:.4f}）")
    else:
        print(f"罚参数：固定 α = {cfg.alpha}")
    print(f"HJB 完整步：Adam {cfg.hjb_adam_steps} + L-BFGS {cfg.hjb_lbfgs_blocks} 块 | "
          f"廉价步：Adam {cfg.cheap_adam_steps} + L-BFGS {cfg.cheap_lbfgs_blocks} 块 | "
          f"每 {cfg.full_step_every} 步一次完整校正")
    print(f"力学重解：Adam {cfg.mech_adam_steps} + L-BFGS {cfg.mech_lbfgs_blocks} 块"
          f"（热启动，max|Δφ| < {cfg.mech_skip_tol} 时跳步）")
    print(f"HJB 模块：{hjb_mod.__name__} | FEM 对拍：{cfg.fem_check}")
    print("=" * 72)

    history: List[Dict] = []
    force_full = True   # 第 1 步必为完整校正步
    t_start = time.time()

    for k in range(1, cfg.n_iter + 1):
        # 加速机制 1：完整校正步 / 廉价预测步
        cheap = not force_full and (k % cfg.full_step_every != 0)
        force_full = False
        mode = "廉价预测步" if cheap else "完整校正步"
        alpha_used = alpha_cur
        print("\n" + "#" * 72)
        print(f"### 迭代 {k}/{cfg.n_iter}（{mode}）：t ∈ [{(k - 1) * cfg.dt}, {k * cfg.dt}]"
              f" | α = {alpha_used:.4g}")
        print("#" * 72)

        # Step B：HJB 演化（冻结 mech_{k-1}，phi_{k-1} → phi_k）
        state = run_hjb_step(k, cfg, cheap, alpha_used)

        # 廉价步护栏：真实 HJB 残差超标则下一步强制完整校正
        if cheap and cfg.cheap_residual_tol is not None:
            rv = state.get("residual_rms_val")
            if rv is not None and rv > cfg.cheap_residual_tol:
                print(f"[护栏] 廉价步 HJB 残差 RMS(val) = {rv:.2e} 超过阈值 "
                      f"{cfg.cheap_residual_tol:.2e}，下一步改为完整校正步")
                force_full = True

        # 变量传递约定 3：新几何的 ω² 锚点
        omega2_anchor = omega2_anchor_from_state(state, omega2_anchor)

        # Step D：罚参数自适应更新（供下一步使用；ω² 取当步锚点，ΔV 取当步面积）
        if cfg.alpha_adaptive and state.get("area_after") is not None:
            alpha_cur, p_next = update_alpha(state["area_after"], omega2_anchor, cfg)
            print(f"[Step D] ΔV = {state['area_after'] - cfg.v_target:+.4f}，"
                  f"ω² = {omega2_anchor:.4f} → 下一步 α = {alpha_cur:.4g}"
                  f"（罚压力 p = {p_next:.4f}）")

        # 加速机制 2：几何变化量度与力学跳步
        dphi = max_dphi(k, cfg)
        skip_mech = cfg.mech_skip_tol > 0.0 and dphi < cfg.mech_skip_tol

        # Step A：力学重解（phi_k @ t_k，热启动 mech_{k-1} → mech_k）
        t_m0 = time.time()
        if skip_mech:
            # 几何几乎没动：沿用上一接力学的权重，复制保持命名链连续
            for n in MECH_NET_NAMES:
                shutil.copy(os.path.join(cfg.weights_dir, n + mech_suffix(k - 1, cfg)),
                            os.path.join(cfg.weights_dir, n + mech_suffix(k, cfg)))
            mech_diag = None
            print(f"[力学跳步] max|Δφ| = {dphi:.2e} < {cfg.mech_skip_tol:.2e}，"
                  f"沿用 {mech_suffix(k - 1, cfg)}（复制为 {mech_suffix(k, cfg)}）")
        else:
            print(f"[力学重解] max|Δφ| = {dphi:.2e}，ω² 锚点 = {omega2_anchor:.6f}")
            mech_diag = run_mech_solve(k, cfg, omega2_anchor)
        mech_wall = time.time() - t_m0

        record = {
            "iter": k,
            "t": k * cfg.dt,
            "cheap": cheap,
            "alpha_used": alpha_used,
            "p_used": (abs(state["area_before"] - cfg.v_target) / alpha_used
                       if state.get("area_before") is not None else None),
            "max_dphi": dphi,
            "mech_skipped": skip_mech,
            "F_before": state.get("F_before"),
            "F_after": state.get("F_after"),
            "area_before": state.get("area_before"),
            "area_after": state.get("area_after"),
            "omega2_rayleigh_before": state.get("omega2_before"),
            "omega2_rayleigh_after": state.get("omega2_after_rayleigh_frozen_mech"),
            "omega2_fem_after": state.get("omega2_fem_after"),
            "omega2_anchor_used": omega2_anchor,
            "hjb_residual_rms_train": state.get("residual_rms_train"),
            "hjb_residual_rms_val": state.get("residual_rms_val"),
            "grad_norm_zero_level": state.get("grad_norm_near_zero_level"),
            "mech_R": (mech_diag or {}).get("R"),
            "mech_m": (mech_diag or {}).get("m"),
            "mech_res_rms": (mech_diag or {}).get("res_rms"),
            "hjb_wall_s": state.get("wall_time_s"),
            "mech_wall_s": round(mech_wall, 1),
        }
        history.append(record)

        # 每步落盘（崩溃不丢历史）+ 两张汇总图重绘
        cfg_dict = dataclasses.asdict(cfg)
        cfg_dict["dtype"] = str(cfg.dtype)
        with open(cfg.history_out, "w", encoding="utf-8") as f:
            json.dump({"config": cfg_dict, "history": history}, f,
                      ensure_ascii=False, indent=2)
        save_history_figure(history, cfg)
        save_zero_level_figure(k, cfg)

        skip_str = "力学跳步" if skip_mech else f"力学 {mech_wall:.0f}s"
        print(f"[迭代 {k} 完成]（{mode}）F {record['F_before']:.6f} → "
              f"{record['F_after']:.6f} | 面积 → {record['area_after']:.4f}"
              f"（C={cfg.v_target}）| ω²锚点 = {omega2_anchor:.6f} | "
              f"HJB {record['hjb_wall_s']:.0f}s + {skip_str} | "
              f"累计 {time.time() - t_start:.0f}s")

    print("\n" + "=" * 72)
    print(f"[完成] {cfg.n_iter} 次交替迭代，总用时 {time.time() - t_start:.0f}s")
    print(f"  最终 SDF：{os.path.join(cfg.weights_dir, phi_name(cfg.n_iter, cfg))}")
    print(f"  最终力学：{cfg.weights_dir}/*{mech_suffix(cfg.n_iter, cfg)}")
    print(f"  历史记录：{cfg.history_out}")
    print(f"  汇总图：{cfg.loop_fig_dir}/history.png, zero_level_sets.png")
    print("=" * 72)
    return history


if __name__ == "__main__":
    main()
