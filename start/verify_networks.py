# -*- coding: utf-8 -*-
"""
网络初始化自检脚本（对应《网络初始化说明.md》的规格逐条验证）

验证内容：
  1. 网络数量与名字（5 力学 + 1 SDF）
  2. 参数量：力学网络 8577（~8.5K），SDF 网络 8641（~8.6K）
  3. 所有参数与 buffer 均为 float64
  4. 内置归一化：域四角映射到 [-1, 1]^2 的角点；SDF 的 t 通道不归一化
  5. Xavier 正态初始化（权重非零、偏置全零）
  6. 输出形状 (N, 1)
  7. 自动微分给出对物理坐标的导数（与中心差分对比，float64 下应吻合到 ~1e-8）
  8. 保存 / 加载往返一致（含归一化 buffer）

运行：python verify_networks.py
全部通过时退出码为 0，否则为 1。
"""

import os
import sys
import tempfile

import torch

from networks import (
    MECH_NET_NAMES,
    SDF_NET_NAME,
    NetworkConfig,
    build_all_networks,
    count_parameters,
    load_networks,
    save_networks,
)

EXPECTED_PARAMS = {"mech": 8577, "sdf": 8641}

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    torch.manual_seed(0)
    cfg = NetworkConfig()
    bundle = build_all_networks(cfg)
    mech, phi = bundle.mechanics, bundle.phi

    # 1. 网络数量与名字
    check(
        "网络数量与名字",
        list(mech.keys()) == MECH_NET_NAMES and SDF_NET_NAME in bundle.all_modules(),
        f"mechanics={list(mech.keys())}, sdf='{SDF_NET_NAME}'",
    )

    # 2. 参数量
    n_mech = count_parameters(mech[MECH_NET_NAMES[0]])
    n_sdf = count_parameters(phi)
    check(
        "参数量（力学 8577 / SDF 8641）",
        n_mech == EXPECTED_PARAMS["mech"]
        and n_sdf == EXPECTED_PARAMS["sdf"]
        and all(count_parameters(m) == n_mech for m in mech.values()),
        f"mech={n_mech}, sdf={n_sdf}",
    )

    # 3. float64
    all64 = all(
        p.dtype == torch.float64 for net in bundle.all_modules().values() for p in net.parameters()
    ) and all(
        b.dtype == torch.float64 for net in bundle.all_modules().values() for b in net.buffers()
    )
    check("全部参数与 buffer 为 float64", all64)

    # 4. 内置归一化
    ux = mech["u_x"]
    corners = torch.tensor(
        [[0.0, 0.0], [cfg.lx, 0.0], [0.0, cfg.ly], [cfg.lx, cfg.ly]], dtype=torch.float64
    )
    normalized = (corners - ux.shift) * ux.scale
    check(
        "力学网络归一化：域四角 -> [-1,1]^2 角点",
        torch.allclose(normalized.abs(), torch.ones_like(normalized), atol=1e-15),
        f"shift={ux.shift.tolist()}, scale={ux.scale.tolist()}",
    )
    check(
        "SDF 网络 t 通道不归一化",
        phi.shift[2].item() == 0.0 and phi.scale[2].item() == 1.0,
        f"shift={phi.shift.tolist()}, scale={phi.scale.tolist()}",
    )

    # 5. Xavier 初始化：偏置全零、权重非零
    bias_zero = all(
        torch.all(m.bias == 0)
        for net in bundle.all_modules().values()
        for m in net.net
        if isinstance(m, torch.nn.Linear)
    )
    weight_nonzero = all(
        torch.any(m.weight != 0)
        for net in bundle.all_modules().values()
        for m in net.net
        if isinstance(m, torch.nn.Linear)
    )
    check("Xavier 初始化（偏置全零、权重非零）", bias_zero and weight_nonzero)

    # 6. 输出形状 (N, 1)
    xy = torch.rand(7, 2, dtype=torch.float64) * torch.tensor([cfg.lx, cfg.ly])
    xyt = torch.cat([xy, torch.zeros(7, 1, dtype=torch.float64)], dim=1)
    shapes_ok = all(m(xy).shape == (7, 1) for m in mech.values()) and phi(xyt).shape == (7, 1)
    check("输出形状 (N, 1)", shapes_ok)

    # 7. 自动微分 = 对物理坐标的导数（与中心差分对比）
    pt = torch.tensor([[0.5, 0.2]], dtype=torch.float64, requires_grad=True)
    val = mech["u_x"](pt)
    (grad,) = torch.autograd.grad(val, pt)
    eps = 1e-6
    fd = torch.zeros_like(grad)
    for i in range(2):
        dp = pt.detach().clone()
        dp[0, i] += eps
        dm = pt.detach().clone()
        dm[0, i] -= eps
        fd[0, i] = (mech["u_x"](dp) - mech["u_x"](dm)).item() / (2 * eps)
    err = (grad - fd).abs().max().item()
    check(
        "自动微分给出物理坐标导数（对比中心差分）",
        err < 1e-8,
        f"autodiff={grad.tolist()[0]}, 中心差分={fd.tolist()[0]}, max|err|={err:.2e}",
    )

    # 8. 保存 / 加载往返一致
    with tempfile.TemporaryDirectory() as tmp:
        paths = save_networks(bundle, tmp)
        names = sorted(os.path.basename(p) for p in paths)
        bundle2 = load_networks(tmp, cfg)
        with torch.no_grad():
            same = all(
                torch.equal(mech[k](xy), bundle2.mechanics[k](xy)) for k in MECH_NET_NAMES
            ) and torch.equal(phi(xyt), bundle2.phi(xyt))
    check(
        "保存 / 加载往返一致（6 个 *_init.pt）",
        same and names == sorted(n + cfg.save_suffix for n in MECH_NET_NAMES + [SDF_NET_NAME]),
        f"files={names}",
    )

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} 项通过" + ("，全部通过。" if n_fail == 0 else f"，{n_fail} 项失败。"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
