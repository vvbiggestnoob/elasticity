# HJB 演化：损失函数的数学描述

> 本文档用数学公式完整描述 `start/hjb_step.py` 中实现的损失函数，
> 与《HJB方程优化说明.md》§5、§7 一一对应。修改损失实现时本文档需同步修订。
>
> **2026-09-08 方法修订**：移除外边界损失 L_b（参考论文对 SDF 无域边界条件）；
> 初始条件增加边界带加密点（权重 w_ic = 10）；新增制造解锚点 L_anchor
> 破除 HJB 残差的平解退化。

---

## 1. 记号与演化区间

计算域 $\Omega = [0, L_x]\times[0, L_y]$，$L_x=1.6$，$L_y=0.5$，$|\Omega|=0.8$。
伪时间区间 $t\in[t_{cur},\,t_{new}]$，$t_{cur}=0$，$t_{new}=t_{cur}+dt$。

被训练的对象是 SDF 网络（3×64 tanh，xy 通道内置归一化、t 通道不归一化）：

$$
\tilde\phi(x, y, t) = N_\phi(x, y, t;\,\theta)
$$

材料 = $\{\tilde\phi > 0\}$，材料边界 = 零水平集 $\{\tilde\phi = 0\}$。

**冻结量**（相对 $\theta$ 均为常数，不回传梯度）：

- $\phi_{ref}(x,y,t)$：本次演化前的 SDF 网络（`phi_init.pt` 的冻结副本），
  提供初始条件锚点与"演化前"几何；
- 5 个力学网络 $u_x^{NN}, u_y^{NN}, \sigma_{xx}^{NN}, \sigma_{xy}^{NN}, \sigma_{yy}^{NN}$
  （`*_init.pt`，本阶段不训练）；
- 由 $\phi_{ref}$ 在 $t=t_{cur}$ 切片给出的材料场（$\beta = 0.01$）：

$$
S(\mathbf{x}) = \frac{1}{2}\Bigl(1+\tanh\frac{\phi_{ref}(\mathbf{x},t_{cur})}{2\beta}\Bigr),
\qquad
\rho = S\,\hat\rho,
\qquad
E = E_{\text{void}} + \bigl(E_{\text{solid}}-E_{\text{void}}\bigr)S
$$

其中 $\hat\rho = 100$（重块 $B=[0.76,0.84]\times[0.21,0.29]$ 内）否则 $1$；
$E_{\text{solid}}=1$，$E_{\text{void}}=10^{-6}$，$\nu=0.3$（平面应力）。

## 2. 固定系数场 $V_n$（进入 $\mathcal{L}_r$ 与锚点的速度场）

$V_n$ 在本次演化中**算一次后固定**（说明文档 §2）。定义（说明文档 §5）：

$$
\boxed{
V_n(\mathbf{x}) = \underbrace{\boldsymbol{\varepsilon}\!:\!\mathbb{A}\!:\!\boldsymbol{\varepsilon}}_{\text{应变能密度}}
\;-\; \omega^2\,\underbrace{\rho\,\|\mathbf{u}\|^2}_{\text{动能密度}}
\;-\; \frac{1}{\alpha}\Bigl(\int_\Omega S\,d\Omega - C\Bigr)
}
$$

各量的取值口径：

- 应变 $\varepsilon$：冻结位移网络的一阶自动微分（对物理坐标），
  $\varepsilon_{xx}=\partial_x u_x^{NN}$，$\varepsilon_{yy}=\partial_y u_y^{NN}$，
  $\gamma_{xy}=\partial_y u_x^{NN}+\partial_x u_y^{NN}$；
- 应变能密度 $\boldsymbol{\varepsilon}\!:\!\mathbb{A}\!:\!\boldsymbol{\varepsilon}
  = \sigma_{xx}^{u}\varepsilon_{xx} + \sigma_{yy}^{u}\varepsilon_{yy} + \sigma_{xy}^{u}\gamma_{xy}$，
  其中 $\sigma^u$ 由平面应力本构给出（与力学初始化 Rayleigh 商的约定一致）；
- $\omega^2$：冻结力学网络在当前几何上的 Rayleigh 商

$$
\omega^2 = \frac{\displaystyle\int_\Omega S\,\bigl(\boldsymbol{\varepsilon}\!:\!\mathbb{A}\!:\!\boldsymbol{\varepsilon}\bigr)\,d\Omega}
{\displaystyle\int_\Omega S\,\hat\rho\,\|\mathbf{u}\|^2\,d\Omega}
$$

- $\int_\Omega S\,d\Omega$：当前面积；$C = 0.4$（50% 体积分数）；$\alpha = 1.0$（本阶段固定）。

两个积分量（$\omega^2$、面积）在 65536 点 Sobol 积分点集上估计一次；
$V_n$ 随后在各训练/验证点集上**逐点预计算为常数张量**，训练中查表使用。
孔洞区内 $E\approx 0$、$\rho\approx 0$，前两项自然消失，$V_n$ 退化为罚项常数。

物理方向：$V_n > 0$ → 边界沿材料外法向外扩（加材料）；$V_n < 0$ → 内缩（去材料）。

## 3. 各项损失

所有求和均为采样点上的蒙特卡洛均值，**均不乘材料掩码 $S$**
（SDF 必须在整个设计域上演化，否则孔洞内无法生成新边界；说明文档 §7）。

点集：

- 内部点 $\{(\mathbf{x}_i, t_i)\}_{i=1}^{N_r}$：$(x,y,t)$ 时空 3D Sobol 采样，
  $t_i\sim U(t_{cur}, t_{new})$；
- 初始条件点 $\{\mathbf{x}_j\}_{j=1}^{N_0}$：内部 $(x,y)$ 2D Sobol（$w_j=1$）
  + **边界带加密点**（$w_j = w_{ic} = 10$），$t\equiv t_{cur}$。

**(1) HJB 残差**（内部点，驱动零水平集以速度 $V_n$ 运动）：

$$
\mathcal{L}_{r} = \frac{1}{N_r}\sum_{i=1}^{N_r}
\Bigl|\,\frac{\partial\tilde\phi}{\partial t}(\mathbf{x}_i, t_i)
\;-\; V_n(\mathbf{x}_i)\,\bigl|\nabla\tilde\phi(\mathbf{x}_i, t_i)\bigr|\,\Bigr|^2
$$

- $\nabla\tilde\phi$ 仅含空间分量 $(\partial_x,\partial_y)$；
  $\bigl|\nabla\tilde\phi\bigr| = \sqrt{\tilde\phi_x^2+\tilde\phi_y^2}$（实现中加 $10^{-30}$ 保护）；
- $\partial\tilde\phi/\partial t$ 与 $\nabla\tilde\phi$ 由对输入 $(x,y,t)$ 的一阶自动微分一次给出；
- **符号**：残差取 $\tilde\phi_t - V_n|\nabla\tilde\phi|$，与参考论文式 (5) 及 §5 的
  $V_n$ 物理方向自洽（"$V_n>0$ 外扩"）；说明文档 §4 原方程的 "+" 号为笔误，
  勘误见《HJB方程优化说明.md》§4/§11 与《HJB训练代码使用文档.md》§5.1。

**(2) 初始条件**（$t=t_{cur}$ 切片锚定当前 SDF，逐点加权）：

$$
\mathcal{L}_{0} = \frac{\displaystyle\sum_{j=1}^{N_0} w_j\,
\Bigl|\,\tilde\phi(\mathbf{x}_j, t_{cur}) - \phi_{ref}(\mathbf{x}_j, t_{cur})\,\Bigr|^2}
{\displaystyle\sum_{j=1}^{N_0} w_j}
$$

目标值逐点预计算（$\phi_{ref}$ 冻结前向）。热启动下初始时 $\mathcal{L}_0 = 0$。

**边界带加密点**（$w_j = w_{ic} = 10$）：沿四边内侧窄带采样（带宽 $4\beta = 0.04$，
按边长比例分配，每边内隔点严格压在边上 $d=0$，外加 4 个角点），与
《SDF初始化说明》的边界带同策略。作用：锚定 rim 处的 $\phi$——纯内部 Sobol
采样下 rim 是零测集，光滑网络会把边界拐角（$\phi=0$ 的最小值正在边界上）
抹圆，rim 处 $\phi$ 系统性虚增（实测 +0.077），$S$ 由 0.5 抬到 ~1、面积失真。

**(3) 制造解锚点**（2026-09-08 新增，可选，$\lambda_{anchor}$ 默认 1，置 0 关闭）：

$$
\mathcal{L}_{anchor} = \frac{1}{N_r}\sum_{i=1}^{N_r}
\Bigl|\,\tilde\phi(\mathbf{x}_i, t_i) - \underbrace{\Bigl[\phi_{ref}(\mathbf{x}_i, t_{cur})
+ (t_i - t_{cur})\,V_n(\mathbf{x}_i)\,\bigl|\nabla\phi_{ref}(\mathbf{x}_i, t_{cur})\bigr|\Bigr]}_{\text{一阶制造解（预计算常数）}}\Bigr|^2
$$

- $|\nabla\phi_{ref}|$ 由冻结 $\phi_{ref}$ 的一阶自动微分预计算（`phi_ref_val_and_gradnorm()`）；
- **动机：HJB 残差存在平解退化**——常函数 $\tilde\phi\equiv const$ 的残差恒为零
  （$\tilde\phi_t=0$、$|\nabla\tilde\phi|=0$），残差只约束比值
  $\tilde\phi_t/|\nabla\tilde\phi| = V_n$，$\tilde\phi$ 的**值**完全靠初始条件钉；
  单靠 $\mathcal{L}_0$ 无法把剖面钉在整个时间柱面上。制造解（一阶 Taylor）
  给出显式剖面目标，从根上破除退化；$dt$ 小（0.005）时一阶制造解与真解几乎一致，
  且与网格对照版（`hjb_fem.py`）的 Godunov 推进在一阶意义下等价。

## 4. 总损失

$$
\boxed{
\mathcal{L}_{HJB} = \lambda_r\,\mathcal{L}_r + \lambda_0\,\mathcal{L}_0
+ \lambda_{anchor}\,\mathcal{L}_{anchor},
\qquad \lambda_r = \lambda_0 = \lambda_{anchor} = 1
}
$$

（$\lambda_{anchor} = 0$ 时退化为 $\mathcal{L}_r + \mathcal{L}_0$ 两项纯 PINN。）

## 5. 实现备注

- **蒙特卡洛口径**：各项均值 $\frac{1}{N}\sum(\cdot)$ 近似 $\frac{1}{|\Omega|}\int_\Omega(\cdot)\,d\Omega$，
  常数因子不影响优化。$\mathcal{L}_0$ 为加权均值（除以 $\sum w_j$），
  边界带权重只改变点间相对重要性。
- **梯度图**：训练时一阶导数以 `create_graph=True` 计算（损失对 $\theta$ 可导，
  即双重反向传播）；验证/诊断时 `create_graph=False` 只取值。$V_n$、初始条件
  目标、制造解目标均为 detach 常数，不参与梯度。
- **采样规模**：Adam 点池 $N_r$ 池 = 20000（每步取 2000）、$N_0$ 合并池 =
  10000 内部 + 4000 边界带（每步取 1000）；L-BFGS 固定 8192 + 4096 + 1024(带)；
  独立验证集 4096 + 2048 + 512(带)（不同种子，只评估不训练；默认仅打印观察，
  不用于早停/回滚）。
- **已移除：外边界损失 $\mathcal{L}_b$**（2026-09-08）。原为四边零法向梯度
  $|\partial\tilde\phi/\partial n|^2$。参考论文对 SDF 不施加域边界条件；
  且 $\mathcal{L}_b$ 与 $\mathcal{L}_0$ 在 $t=0$ 棱线上冲突（初始 SDF 在边界上
  $|\partial\phi/\partial n|=1$），零 Neumann 条件还会把边界值钉死、使零水平集
  无法从域边界后退。移除后边界为出流式（值由 PDE + 内部信息决定），可自由后退。
- **典型量级**（冒烟实测）：初始 $\mathcal{L}_r\sim O(1)$（$V_n$ 在重块处达 $-25$，
  且 t 通道未训练时 $\partial\tilde\phi/\partial t$ 为随机小量）、
  $\mathcal{L}_0 = 0$（热启动）、$\mathcal{L}_{anchor}$ 初始为 t 通道随机外推的
  误差；训练后各项降至 $10^{-2}\sim10^{-3}$ 以下。
- **目标函数只用于诊断**：$F = -\omega^2 + \frac{1}{2\alpha}\bigl(\int_\Omega S\,d\Omega - C\bigr)^2$
  不进入训练损失；$V_n$ 已含其最速下降方向的信息，训练后检查 $F$ 应下降（说明文档 §9）。
