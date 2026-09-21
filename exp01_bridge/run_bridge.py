# -*- coding: utf-8 -*-
"""run_bridge.py — exp01_bridge：角部硬非设计域 + PINN+HJB 交替循环实验。

目标：复现传统方法（E:\\Desktop\\towangbt，PCLS+FEM）的桥式/框架结构——
四角饱满（拱脚）、上下边中段开槽，而不是角部被侵蚀的透镜。

做法：在原有 PINN+HJB（start/）基础上，给 HJB 演化加角部硬非设计域
（hjb_step_eik.py 的 corner_freeze，2026-09-14 新增）：四角 corner_freeze 见方
区域 V_n 置 0 且 φ 锚定到演化前形状，强制保留角部材料。不强制对称。

用法（位置参数均可选）：
  python run_bridge.py [n_iter] [corner_freeze] [dt] [mech_adam] [mech_lbfgs] [hjb_adam] [hjb_lbfgs]
默认：n_iter=80, corner_freeze=0.08, dt=0.005, mech 1200/50, hjb 1500/100
产物全部落在本目录（weights/ 与 loop_figures/），不污染 start/。
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
START_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "start")
sys.path.insert(0, START_DIR)

import alternating_loop as al


def _arg(i, default, cast):
    return cast(sys.argv[i]) if len(sys.argv) > i else default


n_iter = _arg(1, 80, int)
corner_freeze = _arg(2, 0.08, float)
dt = _arg(3, 0.015, float)
mech_adam = _arg(4, 1200, int)
mech_lbfgs = _arg(5, 50, int)
hjb_adam = _arg(6, 1500, int)
hjb_lbfgs = _arg(7, 100, int)

cfg = al.LoopConfig(
    n_iter=n_iter,
    dt=dt,
    corner_freeze=corner_freeze,
    w_corner_freeze=10.0,
    # 初始条件权重（默认 0.1 → 10）：修复边界 φ 系统性膨胀/回填
    # （每步处方侵蚀量小，弱锚定下被边界拟合误差淹没；消融 diag_lam0_sweep.py：
    #   lam_0 增大单调改善，lam_0=10 时面积 −0.0028/步、左边界缺口守住、
    #   顶边正常侵蚀未过度锚定）。2026-09-20。
    lam_0=10.0,
    # 训练量（热启动，较默认值适当压缩以控制单步耗时）
    mech_adam_steps=mech_adam,
    mech_lbfgs_blocks=mech_lbfgs,
    hjb_adam_steps=hjb_adam,
    hjb_lbfgs_blocks=hjb_lbfgs,
    # 产物全部落在本实验目录
    weights_dir=os.path.join(SCRIPT_DIR, "weights"),
    loop_fig_dir=os.path.join(SCRIPT_DIR, "loop_figures"),
    history_out=os.path.join(SCRIPT_DIR, "weights", "loop_history.json"),
)

if __name__ == "__main__":
    print(f"exp01_bridge：角部硬非设计域边长 = {corner_freeze}，n_iter = {n_iter}，"
          f"dt = {dt}（不强制对称）| mech {mech_adam}/{mech_lbfgs}，hjb {hjb_adam}/{hjb_lbfgs}")
    al.main(cfg)
