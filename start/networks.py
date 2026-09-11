# -*- coding: utf-8 -*-
"""
网络初始化实现（对应《网络初始化说明.md》，唯一权威说明）

共 6 个全连接 MLP 网络，分两组：
  - 力学网络（5 个）：u_x, u_y, sigma_xx, sigma_xy, sigma_yy，输入 (x, y)，输出标量
  - SDF 网络（1 个）：phi，输入 (x, y, t)，输出标量（t 为伪时间维，初始化阶段恒取 0）

统一规格：
  - 隐藏层 3 层 x 64 神经元，tanh 激活，线性输出层（1 维）
  - Xavier 正态初始化，偏置置 0
  - 第一层之前内置固定（不可训练）仿射归一化，把物理坐标 (x, y) 映射到 [-1, 1]^2；
    SDF 网络的 t 通道不归一化
  - 全程 float64（L-BFGS 强 Wolfe 线搜索需要双精度）

对外接口始终是物理坐标：归一化在前向传播内部完成，自动微分经链式法则穿过该固定
仿射层，得到的仍是对物理坐标的导数，损失公式无需改写。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# 力学网络名称（固定顺序，保存/加载时按此命名）
MECH_NET_NAMES: List[str] = ["u_x", "u_y", "sigma_xx", "sigma_xy", "sigma_yy"]
SDF_NET_NAME: str = "phi"


@dataclass
class NetworkConfig:
    """网络结构与几何参数。

    lx, ly 为设计域尺寸（物理坐标范围 x in [0, lx], y in [0, ly]），
    用于内置归一化仿射；其余为 MLP 结构参数。
    """

    lx: float = 1.6
    ly: float = 0.5
    hidden_layers: int = 3
    hidden_width: int = 64
    dtype: torch.dtype = torch.float64
    # 各网络文件名后缀（如保存为 u_x_init.pt），由 save_networks 使用
    save_suffix: str = "_init.pt"


@contextmanager
def _default_dtype(dtype: torch.dtype):
    """临时切换 torch 默认 dtype，保证初始化直接在目标精度下进行。"""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


class NormalizedMLP(nn.Module):
    """内置固定输入归一化的全连接 MLP（tanh）。

    前向传播：y = MLP((x - shift) * scale)
    其中 shift / scale 是固定 buffer（不可训练、随 state_dict 保存），
    把物理坐标逐维仿射到目标区间；对输入求导时链式法则自动穿过，
    因此 torch.autograd.grad 给出的仍是对物理坐标的导数。
    """

    def __init__(
        self,
        in_dim: int,
        shift: List[float],
        scale: List[float],
        hidden_layers: int = 3,
        hidden_width: int = 64,
    ) -> None:
        super().__init__()
        assert len(shift) == in_dim and len(scale) == in_dim

        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(d, hidden_width), nn.Tanh()]
            d = hidden_width
        layers.append(nn.Linear(d, 1))  # 线性输出层，1 维标量
        self.net = nn.Sequential(*layers)

        # 固定仿射归一化参数：persistent=True 使其随 state_dict 一起保存，
        # 保证加载权重时归一化定义与训练时一致
        self.register_buffer("shift", torch.tensor(shift))
        self.register_buffer("scale", torch.tensor(scale))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier 正态初始化权重，偏置置 0（含输出层）。"""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim) 物理坐标 -> (N, 1) 标量输出。"""
        return self.net((x - self.shift) * self.scale)


def _xy_shift_scale(cfg: NetworkConfig) -> Dict[str, List[float]]:
    """x~ = 2x/lx - 1, y~ = 2y/ly - 1  =>  (x - lx/2) * (2/lx)。"""
    return {"shift": [cfg.lx / 2.0, cfg.ly / 2.0], "scale": [2.0 / cfg.lx, 2.0 / cfg.ly]}


def build_mechanics_networks(cfg: Optional[NetworkConfig] = None) -> nn.ModuleDict:
    """创建 5 个力学网络（输入 (x, y)，输出标量），返回 ModuleDict。

    键为 MECH_NET_NAMES 中的名字：u_x, u_y, sigma_xx, sigma_xy, sigma_yy。
    """
    cfg = cfg or NetworkConfig()
    ss = _xy_shift_scale(cfg)
    with _default_dtype(cfg.dtype):
        nets = nn.ModuleDict(
            {
                name: NormalizedMLP(
                    in_dim=2,
                    shift=ss["shift"],
                    scale=ss["scale"],
                    hidden_layers=cfg.hidden_layers,
                    hidden_width=cfg.hidden_width,
                )
                for name in MECH_NET_NAMES
            }
        )
    return nets.to(cfg.dtype)


def build_sdf_network(cfg: Optional[NetworkConfig] = None) -> NormalizedMLP:
    """创建 SDF 网络 phi（输入 (x, y, t)，输出标量）。

    (x, y) 通道归一化到 [-1, 1]，t 通道不归一化（初始化恒为 0，演化阶段量级小）。
    """
    cfg = cfg or NetworkConfig()
    ss = _xy_shift_scale(cfg)
    with _default_dtype(cfg.dtype):
        net = NormalizedMLP(
            in_dim=3,
            shift=ss["shift"] + [0.0],
            scale=ss["scale"] + [1.0],
            hidden_layers=cfg.hidden_layers,
            hidden_width=cfg.hidden_width,
        )
    return net.to(cfg.dtype)


@dataclass
class NetworkBundle:
    """6 个网络的集合：mechanics（ModuleDict，5 个力学网络）+ phi（SDF 网络）。"""

    mechanics: nn.ModuleDict
    phi: NormalizedMLP
    config: NetworkConfig = field(default_factory=NetworkConfig)

    def all_modules(self) -> Dict[str, nn.Module]:
        """按名字返回全部 6 个网络（力学 5 个 + phi）。"""
        d: Dict[str, nn.Module] = dict(self.mechanics.items())
        d[SDF_NET_NAME] = self.phi
        return d


def build_all_networks(cfg: Optional[NetworkConfig] = None) -> NetworkBundle:
    """一次性创建全部 6 个网络。"""
    cfg = cfg or NetworkConfig()
    return NetworkBundle(
        mechanics=build_mechanics_networks(cfg),
        phi=build_sdf_network(cfg),
        config=cfg,
    )


def count_parameters(net: nn.Module) -> int:
    """可训练参数总数。"""
    return sum(p.numel() for p in net.parameters())


def save_networks(bundle: NetworkBundle, out_dir: str) -> List[str]:
    """按名字分别保存权重：<out_dir>/<name><suffix>，如 u_x_init.pt、phi_init.pt。

    state_dict 中包含固定归一化 buffer（shift/scale），加载时无需再传几何参数。
    返回写出的文件路径列表。
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, net in bundle.all_modules().items():
        path = os.path.join(out_dir, name + bundle.config.save_suffix)
        torch.save(net.state_dict(), path)
        paths.append(path)
    return paths


def load_networks(
    out_dir: str, cfg: Optional[NetworkConfig] = None, map_location: Optional[str] = None
) -> NetworkBundle:
    """从 <out_dir>/<name><suffix> 加载全部 6 个网络权重，返回 NetworkBundle。"""
    cfg = cfg or NetworkConfig()
    bundle = build_all_networks(cfg)
    for name, net in bundle.all_modules().items():
        path = os.path.join(out_dir, name + cfg.save_suffix)
        state = torch.load(path, map_location=map_location or "cpu", weights_only=True)
        net.load_state_dict(state)
    return bundle
