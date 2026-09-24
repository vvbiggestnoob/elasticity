# -*- coding: utf-8 -*-
"""run_from0.py — exp02_from0：从第 0 步起跑的交替循环（初始化不重做）。

与 exp01_bridge/resume_amax_test.py 从第 10 步续跑相对：本实验从第 0 步几何
（phi_init.pt + 已训练的初始力学 *_init.pt）起跑，sdf_init / mech_init 的
初始化不重做——初始权重直接复制自 exp01_bridge/weights（已训练好的版本，
2026-09-21）。

出图（2026-09-24 起统一风格，见 start/plot_utils.py）：一图一文件、黑白为主、
中间硬块红框标记、四角固定区不标记；按类型分文件夹：
  loop_figures/vn|vn_boundary|energy|kinetic|phi|geometry|delta_S|
              loss_hjb_adam|loss_hjb_lbfgs/iter{k:03d}.png   （HJB 步）
  loop_figures/u_mag|u_x|u_y|sigma_xx|sigma_xy|sigma_yy|
              loss_mech_adam|loss_mech_lbfgs/iter{k:03d}.png （力学重解）
  loop_figures/history/rayleigh.png、area.png、F.png、pressure.png、
              zero_level_sets.png                            （每 10 步与末步重绘）
记录：每步 Rayleigh 商与面积写入 weights/loop_history.json 与
weights/rayleigh_area_history.csv。

用法（位置参数均可选）：
  python run_from0.py [n_iter] [corner_freeze] [dt] [mech_adam] [mech_lbfgs] [hjb_adam] [hjb_lbfgs]
默认：n_iter=200, corner_freeze=0.08, dt=0.015, mech 1200/50, hjb 1500/100
（与 exp01_bridge 最近一次健康运行的配置一致）。
"""
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "start")
sys.path.insert(0, START_DIR)

import alternating_loop as al
from networks import MECH_NET_NAMES

# 已训练初始权重的来源（exp01_bridge/weights；初始化不重做）
INIT_SRC = os.path.join(os.path.dirname(SCRIPT_DIR), "exp01_bridge", "weights")


def _arg(i, default, cast):
    return cast(sys.argv[i]) if len(sys.argv) > i else default


def ensure_init_weights(weights_dir: str):
    """把训练好的初始权重（phi_init.pt + 5 个力学 *_init.pt）复制到本实验目录。"""
    os.makedirs(weights_dir, exist_ok=True)
    need = ["phi_init.pt"] + [n + "_init.pt" for n in MECH_NET_NAMES]
    for f in need:
        d = os.path.join(weights_dir, f)
        if os.path.exists(d):
            continue
        s = os.path.join(INIT_SRC, f)
        if not os.path.exists(s):
            raise FileNotFoundError(
                f"缺少初始权重 {s}；请先把训练好的 {f} 放入 {weights_dir}")
        shutil.copy2(s, d)
        print(f"[初始权重] 复制 {s} -> {d}")


n_iter = _arg(1, 200, int)
corner_freeze = _arg(2, 0.08, float)
dt = _arg(3, 0.015, float)
mech_adam = _arg(4, 1200, int)
mech_lbfgs = _arg(5, 50, int)
hjb_adam = _arg(6, 1500, int)
hjb_lbfgs = _arg(7, 100, int)

cfg = al.LoopConfig(
    n_iter=n_iter,
    k_start=1,                      # 从第 0 步几何起跑（初始化不重做）
    dt=dt,
    corner_freeze=corner_freeze,
    w_corner_freeze=10.0,
    lam_0=10.0,                     # 同 exp01_bridge：修复边界 φ 膨胀/回填
    mech_adam_steps=mech_adam,
    mech_lbfgs_blocks=mech_lbfgs,
    hjb_adam_steps=hjb_adam,
    hjb_lbfgs_blocks=hjb_lbfgs,
    summary_fig_every=10,           # 每 10 步重绘 Rayleigh 商/面积等汇总图
    weights_dir=os.path.join(SCRIPT_DIR, "weights"),
    loop_fig_dir=os.path.join(SCRIPT_DIR, "loop_figures"),
    history_out=os.path.join(SCRIPT_DIR, "weights", "loop_history.json"),
    rayleigh_area_csv=os.path.join(SCRIPT_DIR, "weights",
                                   "rayleigh_area_history.csv"),
)

if __name__ == "__main__":
    ensure_init_weights(cfg.weights_dir)
    print(f"exp02_from0：从第 0 步起跑（初始化不重做），n_iter = {n_iter}，"
          f"corner_freeze = {corner_freeze}，dt = {dt}（不强制对称）| "
          f"mech {mech_adam}/{mech_lbfgs}，hjb {hjb_adam}/{hjb_lbfgs}")
    al.main(cfg)
