# -*- coding: utf-8 -*-
"""alternating_loop 冒烟测试：极小点集与步数，只验证链路连通与变量传递。

Run A：1 次完整迭代（不跳力学），验证命名链 / 伪时间 / 锚点回退 / 热启动。
Run B：3 次迭代，full_step_every=100 + cheap_residual_tol=0 + mech_skip_tol=1，
       验证廉价预测步、残差护栏（k=2 廉价 → k=3 强制完整）、力学跳步复制链。

运行前把初始权重复制到 weights_loop_smoke/（不碰真实 weights/）。
运行后可删除 weights_loop_smoke/、loop_figures_smoke/ 与本文件。
"""
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

SMOKE_W = "weights_loop_smoke"
os.makedirs(SMOKE_W, exist_ok=True)
for f in ["phi_init.pt", "u_x_init.pt", "u_y_init.pt",
          "sigma_xx_init.pt", "sigma_xy_init.pt", "sigma_yy_init.pt"]:
    shutil.copy(os.path.join("weights", f), os.path.join(SMOKE_W, f))

from alternating_loop import LoopConfig, main

HJB_EXTRA = dict(pool_interior=1000, pool_init=500, pool_init_band=100,
                 batch_interior=256, batch_init=64,
                 lbfgs_interior=1000, lbfgs_init=500, lbfgs_init_band=100,
                 lbfgs_max_iter=5,
                 val_interior=500, val_init=200, val_init_band=50,
                 diag_n=2000, fig_nx=161, fig_ny=51)
MECH_EXTRA = dict(pool_interior=1000, pool_block=200, pool_boundary_per_edge=100,
                  batch_interior=256, batch_block=64, batch_boundary_per_edge=32,
                  lbfgs_interior=1000, lbfgs_block=200, lbfgs_boundary_per_edge=50,
                  lbfgs_max_iter=5,
                  val_interior=500, val_block=100, val_boundary_per_edge=50,
                  fig_nx=81, fig_ny=26)

# ----- Run A：完整步 + 力学重解 -----
print("\n>>>>>>>>>> Run A：完整步 + 力学重解 <<<<<<<<<<")
cfg_a = LoopConfig(
    n_iter=1, dt=0.01, fem_check=False, mech_skip_tol=0.0,
    weights_dir=SMOKE_W, loop_fig_dir="loop_figures_smoke",
    history_out=os.path.join(SMOKE_W, "loop_history_a.json"),
    hjb_adam_steps=20, hjb_lbfgs_blocks=1,
    mech_adam_steps=20, mech_lbfgs_blocks=1,
    hjb_extra=HJB_EXTRA, mech_extra=MECH_EXTRA,
)
ha = main(cfg_a)
rec = ha[0]
assert os.path.exists(os.path.join(SMOKE_W, "phi_iter1.pt"))
for n in ["u_x", "u_y", "sigma_xx", "sigma_xy", "sigma_yy"]:
    assert os.path.exists(os.path.join(SMOKE_W, f"{n}_iter1.pt")), f"{n}_iter1.pt 未生成"
assert abs(rec["t"] - 0.01) < 1e-12
assert rec["cheap"] is False and rec["mech_skipped"] is False
assert rec["omega2_fem_after"] is None and rec["omega2_anchor_used"] is not None
print("[Run A 断言通过] 完整步 / 力学重解 / 锚点回退 正常")

# ----- Run B：廉价步 + 护栏 + 力学跳步 -----
print("\n>>>>>>>>>> Run B：廉价步 + 护栏 + 力学跳步 <<<<<<<<<<")
cfg_b = LoopConfig(
    n_iter=3, dt=0.01, fem_check=False,
    full_step_every=100,          # 周期不触发完整步，只靠护栏
    cheap_residual_tol=0.0,       # 任何正残差都触发护栏 → k=3 强制完整
    mech_skip_tol=1.0,            # 强制跳步，测试复制链
    cheap_adam_steps=10, cheap_lbfgs_blocks=1,
    weights_dir=SMOKE_W, loop_fig_dir="loop_figures_smoke",
    history_out=os.path.join(SMOKE_W, "loop_history_b.json"),
    hjb_adam_steps=20, hjb_lbfgs_blocks=1,
    mech_adam_steps=20, mech_lbfgs_blocks=1,
    hjb_extra=HJB_EXTRA, mech_extra=MECH_EXTRA,
)
hb = main(cfg_b)
assert [h["cheap"] for h in hb] == [False, True, False], \
    f"完整/廉价序列错误：{[h['cheap'] for h in hb]}（护栏应使 k=3 回到完整步）"
assert all(h["mech_skipped"] for h in hb), "力学跳步未生效"
for k in (2, 3):
    assert os.path.exists(os.path.join(SMOKE_W, f"phi_iter{k}.pt"))
    for n in ["u_x", "u_y", "sigma_xx", "sigma_xy", "sigma_yy"]:
        assert os.path.exists(os.path.join(SMOKE_W, f"{n}_iter{k}.pt")), \
            f"{n}_iter{k}.pt 未生成（跳步复制链断裂）"
assert abs(hb[-1]["t"] - 0.03) < 1e-12
print("[Run B 断言通过] 廉价步 / 护栏强制完整 / 力学跳步复制链 正常")

print("\n[冒烟断言全部通过]")
