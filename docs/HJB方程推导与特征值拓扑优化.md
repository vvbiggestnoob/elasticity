# HJB 方程推导与特征值拓扑优化

## 一、问题描述

### 弹性特征值方程

$$A\,e(u) = \omega^2 \rho\, u$$

### 边界条件

- $u = 0$，在左端和右端（fixed）
- $\sigma \cdot n = 0$，在顶部和底部（free）
- $\sigma \cdot n = 0$，在孔洞与材料的边界上（free）

### 优化目标

$$\min\; F(\Omega) = -\omega^2(\Omega) + \frac{1}{2\alpha}\left(\int_\Omega dx - C\right)^2$$

其中：
- $\omega^2$：最小特征值（取负号表示最大化）
- $S = H(\text{SDF})$：Heaviside 函数，SDF > 0 时 $S = 1$（有材料），SDF < 0 时 $S = 0$（孔洞）
- $\rho = \hat{\rho} \cdot S$：实际密度 = 材料密度分布 × 有无材料
- $\hat{\rho}$：密度分布（如中心重块密度为 100，其他材料密度为 1，孔洞密度为 0）
- $\alpha$：罚参数
- $C$：目标体积

---

## 二、HJB 方程的推导

### 2.1 水平集方法基础

用隐函数 $\Phi(x, y)$ 表示设计域中的材料分布：

$$\begin{cases} \Phi > 0, & \text{固体区域} \\ \Phi = 0, & \text{边界} \\ \Phi < 0, & \text{孔洞区域} \end{cases}$$

边界不是直接存储的，而是通过 $\Phi = 0$ 的等值面隐式表示。

### 2.2 从边界运动推导 HJB 方程

设边界上一个点 $\boldsymbol{x}(t)$ 随伪时间 $t$ 移动。因为该点始终在边界上：

$$\Phi(\boldsymbol{x}(t), t) = 0$$

对 $t$ 求全导数（链式法则）：

$$\frac{\partial \Phi}{\partial t} + \nabla \Phi \cdot \frac{d\boldsymbol{x}}{dt} = 0$$

边界点沿**法向**移动，速度为 $V_n$：

$$\frac{d\boldsymbol{x}}{dt} = V_n \cdot \boldsymbol{n} = V_n \cdot \frac{\nabla \Phi}{|\nabla \Phi|}$$

代入得：

$$\frac{\partial \Phi}{\partial t} + \nabla \Phi \cdot V_n \frac{\nabla \Phi}{|\nabla \Phi|} = 0$$

$$\frac{\partial \Phi}{\partial t} + V_n \frac{|\nabla \Phi|^2}{|\nabla \Phi|} = 0$$

$$\boxed{\frac{\partial \Phi}{\partial t} + V_n |\nabla \Phi| = 0}$$

这就是 **Hamilton-Jacobi 方程**。

**物理含义**：$\Phi$ 的变化率完全由边界的法向运动速度 $V_n$ 决定。$V_n > 0$ 时边界向外扩张，$V_n < 0$ 时边界向内收缩。

---

## 三、形状导数推导

### 3.1 目标函数的形状导数

对 $F(\Omega) = -\omega^2(\Omega) + \frac{1}{2\alpha}\left(\int_\Omega dx - C\right)^2$ 求形状导数。

#### 第一部分：$-\omega^2$ 的形状导数

由特征值问题的形状灵敏度分析：

$$d(-\omega^2) = -\int_{\partial\Omega}\left(A\,e(u):e(u) - \omega^2\rho|u|^2\right)(V\cdot n)\,dS$$

其中：
- $A\,e(u):e(u)$：**应变能密度**，边界向外扩张时增加的刚度贡献
- $\omega^2\rho|u|^2$：**动能密度**，边界向外扩张时增加的质量贡献
- 两者之差决定了增加材料对特征值的净效应

#### 第二部分：罚项的形状导数

$$d\left[\frac{1}{2\alpha}\left(\int_\Omega dx - C\right)^2\right] = \frac{1}{\alpha}\left(\int_\Omega dx - C\right)\int_{\partial\Omega}(V\cdot n)\,dS$$

链式法则：外层平方求导得 $2(\int dx - C)$，内层体积的形状导数是 $\int_{\partial\Omega}(V\cdot n)dS$。

#### 合并

$$dF = \int_{\partial\Omega}\left[-\left(A\,e(u):e(u) - \omega^2\rho|u|^2\right) + \frac{1}{\alpha}\left(\int_\Omega dx - C\right)\right](V\cdot n)\,dS$$

记形状梯度密度为 $G$：

$$G = -\left(A\,e(u):e(u) - \omega^2\rho|u|^2\right) + \frac{1}{\alpha}\left(\int_\Omega dx - C\right)$$

则 $dF = \int_{\partial\Omega} G\,(V\cdot n)\,dS$

### 3.2 最速下降方向

为了让 $dF < 0$（目标函数下降），选负梯度方向：

$$V_n = -G = \left(A\,e(u):e(u) - \omega^2\rho|u|^2\right) - \frac{1}{\alpha}\left(\int_\Omega dx - C\right)$$

这样 $dF = -\int_{\partial\Omega}G^2\,dS < 0$ ✓

**物理含义**：
- $A\,e(u):e(u)$ 大 → 该处材料对刚度贡献大 → $V_n > 0$ → 边界向外扩张（保留材料）
- $\omega^2\rho|u|^2$ 大 → 该处材料对质量贡献大 → 降低特征值 → $V_n < 0$ → 边界收缩（去除材料）
- 罚项：当前体积偏大时 $\int dx - C > 0$，$V_n$ 减小 → 收缩

### 3.3 代入 HJB 方程

$$\frac{\partial\phi}{\partial t} + V_n|\nabla\phi| = 0$$

代入 $V_n$：

$$\frac{\partial\phi}{\partial t} + \left[\left(A\,e(u):e(u) - \omega^2\rho|u|^2\right) - \frac{1}{\alpha}\left(\int_\Omega dx - C\right)\right]|\nabla\phi| = 0$$

> **注意**：标准 HJB 方程中是 $|\nabla\phi|$（不是 $|\nabla\phi|^2$）。因为 $n\cdot\nabla\phi = \frac{\nabla\phi}{|\nabla\phi|}\cdot\nabla\phi = |\nabla\phi|$。

### 3.4 初始条件

$$\phi(0) = \phi_0$$

### 3.5 罚参数自适应更新

$$\frac{1}{\alpha^{(k+1)}} = \min\left(\frac{\omega^2\,\tau}{\left(\int_\Omega dx - C\right)^2},\ \alpha_{max}\right)$$

- 当体积偏差大时 → $1/\alpha$ 受限于 $\alpha_{max}$ → 罚项强 → 优先满足体积约束
- 当体积接近目标时 → $1/\alpha$ 由 $\omega^2\tau/(\Delta V)^2$ 控制 → 罚项适中 → 更关注特征值优化
- $\tau$ 是缩放因子（可取 1.1）

---

## 四、DNN 水平集方法

### 4.1 核心思想

用神经网络替代网格上的 $\Phi$：

$$\Phi(x, y, t) = \mathbb{N}(x, y, \boldsymbol{\theta}(t))$$

对 $t$ 求导（链式法则）：

$$\frac{\partial \Phi}{\partial t} = \frac{\partial \mathbb{N}}{\partial \boldsymbol{\theta}} \cdot \frac{d\boldsymbol{\theta}}{dt}$$

代入 HJB 方程：

$$\frac{\partial \mathbb{N}}{\partial \boldsymbol{\theta}} \cdot \frac{d\boldsymbol{\theta}}{dt} = -V_n |\nabla \mathbb{N}|$$

### 4.2 转化为 ODE

这是一个超定方程组（空间点很多，但网络参数共享）。用 **Moore-Penrose 伪逆**求最小二乘解：

$$\frac{d\boldsymbol{\theta}}{dt} = -\mathcal{M}^+ \cdot V_n \cdot |\nabla \mathbb{N}|$$

其中：

$$\mathcal{M}^+ = \left(\left(\frac{\partial\mathbb{N}}{\partial\boldsymbol{\theta}}\right)^T \frac{\partial\mathbb{N}}{\partial\boldsymbol{\theta}}\right)^{-1} \left(\frac{\partial\mathbb{N}}{\partial\boldsymbol{\theta}}\right)^T$$

**结果**：PDE 转化为关于网络参数 $\boldsymbol{\theta}$ 的 **ODE**，用 RKF45 方法求解。

### 4.3 重初始化

每个迭代步对 SDF 网络进行重初始化，保持 $|\nabla\Phi| \approx 1$：

$$\frac{\partial\mathbb{N}}{\partial\theta} \cdot \frac{\partial\theta}{\partial t} = \text{sign}(\Phi_{\text{initial}})(|\nabla\mathbb{N}| - 1)$$

---

## 五、交替迭代算法流程

```
┌─────────────────────────────────────────────────────────┐
│                   外层循环（拓扑优化）                      │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step 1: 用 SDF 网络计算几何                       │  │
│  │  Φ = N_sdf(x, y, θ)                               │  │
│  │  S = H(Φ)  →  ρ = ρ̂ · S                          │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step 2: 用 5 个网络求解弹性特征值问题              │  │
│  │  应力网络 + 应变网络 → 位移场 u                     │  │
│  │  → 计算 ω²                                        │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step 3: 灵敏度分析 → 计算 V_n                     │  │
│  │  V_n = (Ae(u):e(u) - ω²ρ|u|²) - (1/α)(∫dx - C)  │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step 4: 用 HJB 方程更新 SDF 网络参数 θ             │  │
│  │  dθ/dt = -M⁺ · V_n · |∇N_sdf|                    │  │
│  │  用 RKF45 求解这个 ODE                             │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step 5: 重初始化 SDF 网络                         │  │
│  │  保持 |∇Φ| ≈ 1                                   │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         ↓                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step 6: 更新罚参数 α                              │  │
│  │  1/α^(k+1) = min(ω²τ/(∫dx-C)², α_max)           │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         ↓                                │
│                  返回 Step 1，直到收敛                     │
└─────────────────────────────────────────────────────────┘
```

---

## 六、6 个网络的分工

| 网络 | 作用 | 输入 | 输出 |
|------|------|------|------|
| 网络 1-5 | 表示应力/应变分量 | $(x, y)$ | $\sigma_{xx}, \sigma_{yy}, \sigma_{xy}, \varepsilon_{xx}, \varepsilon_{yy}$ 等 |
| 网络 6 | 表示 SDF | $(x, y)$ | $\Phi$（标量） |

**交替迭代**：
1. **固定 SDF 网络** → 用网络 1-5 求解力学问题 → 得到位移、应力、应变、$\omega^2$
2. **固定网络 1-5** → 用灵敏度计算 $V_n$ → 用 HJB 方程更新 SDF 网络参数
3. 重复直到收敛

---

## 七、推导验证总结

| 步骤 | 内容 | 状态 |
|------|------|------|
| 目标函数 | $-\omega^2$ + 二次罚 | ✅ |
| 形状导数 | Hadamard 公式 | ✅ |
| 最速下降 | $V_n = -G$ | ✅ |
| HJB 方程 | $\partial\phi/\partial t + V_n\|\nabla\phi\| = 0$ | ✅（注意是 $\|\nabla\phi\|$ 不是 $\|\nabla\phi\|^2$） |
| 罚参数更新 | 自适应策略 | ✅ |
