# -*- coding: utf-8 -*-
"""新出图/记录链路冒烟测试（2026-09-24 重构后）。

验证：
  Run A：k_start=1（从第 0 步起跑），2 次迭代，summary_fig_every=1
    - 类型分文件夹单图：vn/energy/kinetic/phi/geometry/delta_S/loss_hjb_*、
      u_mag/sigma_xx/loss_mech_* 的 iter{001,002}.png
    - history/ 下 rayleigh.png、area.png、F.png、pressure.png、zero_level_sets.png
    - rayleigh_area_history.csv 三行（iter 0/1/2），loop_history.json 含 iter-0 记录
  Run B：k_start=3 续跑 1 步（读 hjb_state_iter2.json 的锚点），命名链衔接

运行后可删除 _smoke_layout/ 与本文件。
"""
import json
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

WORK = "_smoke_layout"
SMOKE_W = os.path.join(WORK, "weights")
FIG = os.path.join(WORK, "loop_figures")
shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(SMOKE_W)

INIT_SRC = os.path.join(os.path.dirname(SCRIPT_DIR), "exp01_bridge", "weights")
for f in ["phi_init.pt", "u_x_init.pt", "u_y_init.pt",
          "sigma_xx_init.pt", "sigma_xy_init.pt", "sigma_yy_init.pt"]:
    shutil.copy(os.path.join(INIT_SRC, f), os.path.join(SMOKE_W, f))

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

# ----- Run A：从第 0 步起跑，2 次迭代 -----
print("\n>>>>>>>>>> Run A：k_start=1，2 次迭代 <<<<<<<<<<")
cfg_a = LoopConfig(
    n_iter=2, k_start=1, dt=0.01, fem_check=False, mech_skip_tol=0.0,
    corner_freeze=0.08, summary_fig_every=1,
    weights_dir=SMOKE_W, loop_fig_dir=FIG,
    history_out=os.path.join(SMOKE_W, "loop_history.json"),
    rayleigh_area_csv=os.path.join(SMOKE_W, "rayleigh_area_history.csv"),
    hjb_adam_steps=20, hjb_lbfgs_blocks=1,
    mech_adam_steps=20, mech_lbfgs_blocks=1,
    hjb_extra=HJB_EXTRA, mech_extra=MECH_EXTRA,
)
ha = main(cfg_a)

for k in (1, 2):
    tag = f"iter{k:03d}"
    assert os.path.exists(os.path.join(SMOKE_W, f"phi_iter{k}.pt"))
    for kind in ("vn", "vn_boundary", "energy", "kinetic", "phi", "geometry",
                 "delta_S", "loss_hjb_adam", "loss_hjb_lbfgs"):
        p = os.path.join(FIG, kind, f"{tag}.png")
        assert os.path.exists(p), f"缺图 {p}"
    for kind in ("u_mag", "u_x", "u_y", "sigma_xx", "sigma_xy", "sigma_yy",
                 "loss_mech_adam", "loss_mech_lbfgs"):
        p = os.path.join(FIG, kind, f"{tag}.png")
        assert os.path.exists(p), f"缺图 {p}"
for name in ("rayleigh.png", "area.png", "F.png", "pressure.png",
             "zero_level_sets.png"):
    p = os.path.join(FIG, "history", name)
    assert os.path.exists(p), f"缺汇总图 {p}"

hist = json.load(open(cfg_a.history_out, encoding="utf-8"))["history"]
assert [h["iter"] for h in hist] == [0, 1, 2], "缺 iter-0 记录"
assert hist[0]["area_after"] is not None and hist[0]["omega2_fem_after"] is not None
assert all(h["omega2_rayleigh_after"] is not None and h["area_after"] is not None
           for h in hist[1:]), "每步 Rayleigh 商/面积记录缺失"

rows = open(cfg_a.rayleigh_area_csv, encoding="utf-8").read().strip().splitlines()
assert len(rows) == 1 + 3, f"CSV 应为表头+3 行，实际 {len(rows)} 行"
assert rows[0] == ("iter,t,omega2_rayleigh_before,omega2_rayleigh_after,"
                   "omega2_fem_after,area_before,area_after")
print("[Run A 断言通过] 分文件夹单图 / history 汇总图 / iter-0 记录 / CSV 正常")

# ----- Run B：k_start=3 续跑 1 步 -----
print("\n>>>>>>>>>> Run B：k_start=3 续跑 <<<<<<<<<<")
cfg_b = LoopConfig(
    n_iter=3, k_start=3, dt=0.01, fem_check=False, mech_skip_tol=0.0,
    corner_freeze=0.08, summary_fig_every=1,
    weights_dir=SMOKE_W, loop_fig_dir=FIG,
    history_out=os.path.join(SMOKE_W, "loop_history_b.json"),
    rayleigh_area_csv=os.path.join(SMOKE_W, "rayleigh_area_b.csv"),
    hjb_adam_steps=20, hjb_lbfgs_blocks=1,
    mech_adam_steps=20, mech_lbfgs_blocks=1,
    hjb_extra=HJB_EXTRA, mech_extra=MECH_EXTRA,
)
hb = main(cfg_b)
assert [h["iter"] for h in hb] == [3]
assert os.path.exists(os.path.join(SMOKE_W, "phi_iter3.pt"))
assert os.path.exists(os.path.join(FIG, "geometry", "iter003.png"))
print("[Run B 断言通过] k_start=3 续跑 / 命名链衔接 正常")

print("\n[冒烟断言全部通过]")
