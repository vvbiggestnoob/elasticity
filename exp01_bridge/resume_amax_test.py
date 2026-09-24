# -*- coding: utf-8 -*-
"""resume_amax_test.py — 从健康检查点（默认 iter9）重启交替循环，测试 Step D 的 α_max。

背景：原运行（dt=0.015，α_max=1e6）在第 13 步崩溃——面积逼近目标 C=0.4 时
罚压力 p = τ·ω²/|ΔV| 随 |ΔV| 反比爆炸（p≈9.84），V_n = (ε:A:ε − ω²ρ|u|²) − p
被均匀侵蚀项淹没，一步把结构抹掉（面积 0.4117 → 0.0434）。

本脚本不改 dt、不给 p 加硬上限，只把 α_max 降到个位数做对比测试。
机制：p(ΔV) = min(τ·ω²/|ΔV|, α_max·|ΔV|)，两支在 |ΔV| = √(τ·ω²/α_max) 处
相交，峰值 p_max = √(τ·ω²·α_max)。dt=0.015 的 CFL 预算 |V_n| ≲ β/dt ≈ 0.67
（β=0.01 为界面厚度）→ α_max ≈ 4 时 p_max ≈ 0.68，恰在预算内。

重启方式：2026-09-24 起 alternating_loop.main 原生支持 k_start（>1 时从
iter{k_start-1} 检查点续跑，首步 α 与 ω² 锚点取自 hjb_state_iter{k_start-1}.json，
迭代编号与命名链无缝衔接），本脚本只剩 α_max 对比的封装。k 由 --kstart 指定，
默认 10（iter9 检查点：面积 0.5619，ω²_FEM = 0.16945）。

用法（位置参数均可选）：
  python resume_amax_test.py [alpha_max] [n_steps] [--iso] [--kstart 10]
默认：alpha_max=4，n_steps=30（约 10 min/步，30 步 ≈ 5 h），kstart=10。

  默认原地运行：iter13+ 的权重/状态/图片覆盖原失败运行的垃圾产物；
  原 loop_history.json 不动，本次历史写 loop_history_resume_amax*.json。
  --iso：把重启所需文件复制到独立目录 weights_amax*/ + loop_figures_amax*/，
  不碰原目录，可同时跑多个 α_max 对比（各测试的逐步图片也各自保留）。
"""
import argparse
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "start")
sys.path.insert(0, START_DIR)

import alternating_loop as al
from networks import MECH_NET_NAMES

# 重启检查点：默认 iter9（面积 0.5619，ω²_FEM = 0.16945），即从第 10 步起跑；
# 崩溃前最后一个健康步是 iter12，要复现原对比测试用 --kstart 13
K_START_DEFAULT = 10


def fmt_g(x):
    """紧凑数字格式（1e6 -> 1e+06，4 -> 4）。"""
    return f"{x:g}"


def print_amax_table(cfg: al.LoopConfig, k_start: int):
    """启动时打印：当前 α_max、个位数候选在同一状态下的 α / p / 理论峰值。

    参考状态取自重启检查点 hjb_state_iter{k_start-1}.json，与首步实际使用的
    area / ω² 锚点一致。p 的两支：远离目标时 p = τ·ω²/|ΔV|（与原公式一致）；
    α_max 起作用后 p = α_max·|ΔV| 随 |ΔV| → 0 自动刹车；
    峰值 p_max = √(τ·ω²·α_max)。
    """
    import json
    state_path = os.path.join(cfg.weights_dir,
                              f"hjb_state_iter{k_start - 1}.json")
    with open(state_path, "r", encoding="utf-8") as f:
        prev = json.load(f)
    area_ref = float(prev["area_after"])
    omega2_ref = float(prev.get("omega2_fem_after")
                       or prev["omega2_after_rayleigh_frozen_mech"])
    beta = al.hjb_mod.HJBConfig().beta
    budget = beta / cfg.dt                      # CFL：|V_n|·dt ≲ β
    dV = abs(area_ref - cfg.v_target)
    inv_uncapped = cfg.tau * omega2_ref / dV ** 2  # 原公式（无 α_max 时）的 1/α
    print("=" * 72)
    print(f"重启检查点 iter{k_start - 1}：面积 = {area_ref:.5f}，"
          f"ΔV = {area_ref - cfg.v_target:+.5f}，ω²_FEM = {omega2_ref:.5f}")
    print("原运行 α_max = 1e6（第 13 步崩溃：p = 9.84 远超 CFL 预算）")
    print(f"CFL 预算：|V_n| ≲ β/dt = {beta}/{fmt_g(cfg.dt)} = {budget:.2f}")
    print(f"{'α_max':>10} | {'首步 α':>10} | {'首步 p':>8} | "
          f"{'峰值 p = √(τ·ω²·α_max)':>22} | 峰值/预算")
    for am in (1, 2, 4, 8, 16, 1e6):
        inv = min(inv_uncapped, am)
        p0 = dV * inv
        p_peak = (cfg.tau * omega2_ref * am) ** 0.5
        mark = "  <== 本次" if am == cfg.alpha_max else ""
        print(f"{fmt_g(am):>10} | {1 / inv:>10.4g} | {p0:>8.4f} | "
              f"{p_peak:>22.3f} | {p_peak / budget:>5.2f}{mark}")
    print("=" * 72)


def preflight(cfg: al.LoopConfig, k_start: int):
    """重启所需文件检查：phi_iter{k-1}.pt、五个力学 *_iter{k-1}.pt、
    hjb_state_iter{k-1}.json（首步 α 与 ω² 锚点的来源）。"""
    k0 = k_start - 1
    need = [al.phi_name(k0, cfg)]
    need += [n + al.mech_suffix(k0, cfg) for n in MECH_NET_NAMES]
    need += [f"hjb_state_iter{k0}.json"]
    missing = [f for f in need
               if not os.path.exists(os.path.join(cfg.weights_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"重启检查点缺文件 {missing}（目录 {cfg.weights_dir}）。")


def setup_iso(cfg: al.LoopConfig, k_start: int, tag: str):
    """--iso：复制重启所需文件到独立目录，改指 cfg 的 weights/fig/history 路径。"""
    src = os.path.join(SCRIPT_DIR, "weights")
    dst = os.path.join(SCRIPT_DIR, f"weights_amax{tag}")
    os.makedirs(dst, exist_ok=True)
    k0 = k_start - 1
    files = [f"hjb_state_iter{k0}.json"]
    files += [al.phi_name(j, cfg) for j in range(0, k0 + 1)]  # 零水平集叠加图要 0..k0
    files += [n + al.mech_suffix(k0, cfg) for n in MECH_NET_NAMES]
    for f in files:
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if not os.path.exists(d) or os.path.getmtime(s) > os.path.getmtime(d):
            shutil.copy2(s, d)
    cfg.weights_dir = dst
    cfg.loop_fig_dir = os.path.join(SCRIPT_DIR, f"loop_figures_amax{tag}")
    cfg.history_out = os.path.join(dst, f"loop_history_amax{tag}.json")
    cfg.rayleigh_area_csv = os.path.join(dst, f"rayleigh_area_amax{tag}.csv")
    print(f"[iso] 独立目录：{dst}（已复制 {len(files)} 个重启文件）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="从健康检查点（默认 iter9）重启，测试 α_max")
    ap.add_argument("alpha_max", type=float, nargs="?", default=4.0,
                    help="罚系数 1/α 上限（默认 4；原运行为 1e6）")
    ap.add_argument("n_steps", type=int, nargs="?", default=30,
                    help="重启后跑的步数（默认 30，约 10 min/步）")
    ap.add_argument("--kstart", type=int, default=K_START_DEFAULT,
                    help=f"重启起始迭代号（默认 {K_START_DEFAULT} = "
                         "从 iter9 检查点出发）")
    ap.add_argument("--iso", action="store_true",
                    help="复制到 weights_amax*/ 独立目录运行，不碰原 weights/")
    args = ap.parse_args()

    tag = fmt_g(args.alpha_max)
    cfg = al.LoopConfig(
        # 与原失败运行完全一致的配置（loop_history.json 的 config 逐项核对），
        # 仅 alpha_max 不同；dt 保持 0.015 不变
        n_iter=args.kstart + args.n_steps - 1,
        k_start=args.kstart,
        dt=0.015,
        corner_freeze=0.08,
        w_corner_freeze=10.0,
        lam_0=10.0,
        alpha_max=args.alpha_max,
        mech_adam_steps=1200,
        mech_lbfgs_blocks=50,
        hjb_adam_steps=1500,
        hjb_lbfgs_blocks=100,
        weights_dir=os.path.join(SCRIPT_DIR, "weights"),
        loop_fig_dir=os.path.join(SCRIPT_DIR, "loop_figures"),
        history_out=os.path.join(SCRIPT_DIR, "weights",
                                 f"loop_history_resume_amax{tag}.json"),
        rayleigh_area_csv=os.path.join(SCRIPT_DIR, "weights",
                                       f"rayleigh_area_resume_amax{tag}.csv"),
    )

    preflight(cfg, args.kstart)
    if args.iso:
        setup_iso(cfg, args.kstart, tag)
        preflight(cfg, args.kstart)               # 复制后再核一次
    else:
        stale = args.kstart + args.n_steps - 1
        print(f"[提示] 原地运行：iter{args.kstart}~{stale} 的权重/图片将覆盖"
              "原失败运行的同号产物；原 loop_history.json 不动。")
    os.makedirs(cfg.loop_fig_dir, exist_ok=True)

    print_amax_table(cfg, args.kstart)
    history = al.main(cfg)

    # 收尾小结：面积轨迹与 p 峰值，快速判断该 α_max 是否收敛且不失稳
    areas = [r["area_after"] for r in history if r.get("area_after") is not None]
    ps = [r["p_used"] for r in history if r.get("p_used") is not None]
    print("\n" + "=" * 72)
    print(f"[测试结束] α_max = {fmt_g(cfg.alpha_max)}，{len(history)} 步")
    if areas:
        print(f"  面积：{areas[0]:.4f} → {areas[-1]:.4f}（目标 {cfg.v_target}，"
              f"最小 {min(areas):.4f}）| p 峰值 = {max(ps):.3f}")
        collapses = [i + 1 for i in range(1, len(areas))
                     if areas[i] < 0.5 * areas[i - 1]]
        if collapses:
            print(f"  [警告] 出现单步面积腰斩（第 {collapses} 个记录步）——"
                  "该 α_max 仍过大")
        else:
            print("  未出现单步面积腰斩。")
    print("=" * 72)
