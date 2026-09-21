# 基于深度神经网络（DNN）的参数化水平集拓扑优化方法

**A Parametric Level Set Method for Topology Optimization based on Deep Neural Network (DNN)**

**作者**：Hao Deng, Albert C. To*

**单位**：匹兹堡大学机械工程与材料科学系，匹兹堡，PA 15261

*通讯作者邮箱：albertto@pitt.edu

---

## 摘要

本文提出了一种基于深度神经网络（DNN）的新的参数化水平集拓扑优化方法。在该方法中，全连接深度神经网络被引入传统水平集方法中，构建了一种有效的结构拓扑优化方法。水平集的隐函数由全连接深度神经网络描述。本文提出了一种基于 DNN 的水平集优化方法，其中 Hamilton-Jacobi 偏微分方程（PDE）被转化为参数化的常微分方程（ODE）。隐函数的零水平集通过更新网络的权重和偏置来更新。参数化重初始化被周期性应用，以防止隐函数在其零水平集附近变得过陡或过平。该方法在最小柔度（minimum compliance）框架下实现，这是拓扑优化领域中一个公认的基准问题。在实践中，设计者希望拥有多个设计方案，以便根据设计经验选择更好的概念设计。基于 DNN 的水平集方法的一个主要优势在于，它能够通过不同的网络架构生成多样且具有竞争力的设计。本文给出了若干数值算例，验证了所提出的基于 DNN 的水平集方法的有效性。

**关键词**：拓扑优化，深度神经网络，水平集方法，多样且具有竞争力的设计

---

## 1. 引言

拓扑优化近年来受到了广泛关注和快速发展，它通过基于梯度的算法在设计域内搜索最优材料分布。Bendsoe 和 Kikuchi 于 1988 年发表了第一个基于均匀化方法的拓扑优化方法 [1]。在过去的二十年中，拓扑优化研究经历了蓬勃发展 [2]。带惩罚的固体各向同性材料（SIMP）方法 [3] 因其有效性和简洁性而在工程中得到了广泛应用。在 SIMP 方法中，人工密度被用来描述材料分布，最优设计通过基于梯度的优化算法获得。然而，最优设计中可能存在中间密度，这会使边界模糊，需要后处理技术来消除灰度区域。事实上，标准的基于密度的方法很难在优化过程中完全消除中间密度 [4]。基于标准密度优化框架，近年来提出了几种先进的方案来实现特征控制、鲁棒设计和长度尺度控制 [5-15]，这些方法能够通过各种投影方法缓解上述问题。近年来还提出了一些其他鲁棒公式 [16-19] 来保证可制造性。总体而言，需要后处理技术来获得边界清晰的最优设计，而后处理后的设计性能相比原始最优设计可能有所退化。

与 SIMP 方法相比，水平集方法是一种"移动边界"方法，在优化过程中演化设计边界，且边界始终清晰 [20]。最初，Osher 和 Sethian [21] 提出水平集方法来处理移动流体的前沿。水平集方法使用隐函数的零水平集表示设计边界，并通过形状灵敏度分析计算速度场，将其代入 Hamilton-Jacobi 偏微分方程（PDE）来演化水平集函数。Osher 和 Sethian [22] 基于投影梯度方法提出了一种用于设计优化问题的水平集方法。该工作随后由 Allaire [23] 和 Wang [24] 进一步发展。Allaire [23] 提出了一种基于形状导数和水平集方法相结合的数值方法用于前沿传播，其中将重量和周长约束作为目标函数。Wang [24] 使用高维标量函数以水平集模型表示结构边界，并建立了速度场与结构灵敏度分析之间的联系。与传统的水平集方法相比，近年来提出了几种参数化水平集方法。Wang 等人 [25] 将径向基函数（RBF）引入传统水平集方法，构建了一种更高效的拓扑优化方法。RBF 水平集优化方法将 Hamilton-Jacobi PDE 转化为参数化 ODE，并通过更新展开系数来更新水平集函数。Peng 和 Wang [26] 提出了分段常数水平集（PCLS）方法来解决形状和拓扑优化问题，其中边界由 PCLS 函数描述。Jiang 等人 [27] 应用基数函数通过单位配点法参数化水平集函数，并引入距离正则化能量泛函来在优化过程中保持所需的符号距离性质。最近，Guo 等人 [28, 29] 提出了一种名为移动可变形组件（MMC）的新计算框架，将移动可变形组件嵌入水平集方案中。该计算方案以显式方式将几何信息和力学信息融入拓扑优化中，并且结构复杂度可以以显式方式轻松控制。另一种称为移动可变形孔洞（MMV）的方法也由 Guo 等人 [30, 31] 提出，该方法引入了一组几何参数以显式方式描述结构边界。近年来，文献中还提出了其他几种先进的参数化水平集方法 [32]。Luo 等人 [33] 提出了一种无需任何灵敏度信息的高效非梯度拓扑优化方法。在其方法中，材料场级数展开（MFSE）被用于参数化几何信息，实现了设计变量的大幅减少，并基于代理模型采用 Kriging 优化算法求解优化问题。

机器学习 [34] 在过去十年中经历了研究兴趣的巨大增长，因为它是构建输入和输出采样数据之间关系的强大工具。随着可用数据的急剧增长和新方法的发展，机器学习已经彻底改变了我们对物理世界的理解，如图像识别 [35]、药物发现 [36] 等。近年来可以找到若干成功应用于物理问题的案例。Raissi 等人 [37] 提出了物理信息神经网络（PINN）用于求解偏微分方程。在其方法中，物理信息神经网络被训练来求解受给定物理定律约束的监督学习任务。基于这一概念，最近提出了几种先进的基于机器学习的方法 [38-41] 用于求解正向和逆向 PDE 问题。近年来也见证了若干将机器学习方法应用于拓扑优化问题的研究。Yu 等人 [42] 提出了一种新颖的基于深度学习的方法，在给定边界条件下无需任何迭代方案即可预测最优设计。Lei 等人 [43] 提出了一种基于机器学习方法实现实时结构拓扑优化的方法。其方法将基于 MMC 的显式框架与支持向量回归（SVR）相结合，建立设计参数与最优设计之间的映射。Oh 和 Jung 等人 [44] 提出了一个将拓扑优化与生成模型（生成对抗网络）以迭代方式集成的框架，用于生成新设计。如文献 [45] 所述，深度神经网络被应用于以学习函数的零水平集表示形状边界，该方法可以表示一整类形状，因此模型大小相比现有工作可以减少一个数量级。在参数化水平集方法的计算框架下，本文提出了一种基于深度学习的参数化水平集方法来实现拓扑优化。当前工作的核心是将深度神经网络引入现有的基于水平集的拓扑优化方法中。隐函数由深度前馈神经网络描述，因此可以保证隐函数的充分光滑性和连续性。

目前，大多数拓扑优化算法旨在寻找能够最小化或最大化目标函数的最优材料分布。在实践中，设计者希望生成多个足够多样且具有竞争力的解，以便他们能够根据审美角度或其他功能需求的经验做出选择。Wang 等人 [46] 通过在 SIMP 方法框架下引入图形多样性约束来实现多样且有竞争力的设计。基于不同的惩罚方法，Yang 等人 [47] 提出了五种简单有效的策略来获得多个解，这些策略被证明能够为设计者提供结构高效且拓扑不同的解。最近，He 和 Xie 等人 [48] 在双向进化结构优化（BESO）框架下提出了三种随机方法来生成多样且有竞争力的设计，其中产生了一系列具有明显不同拓扑的随机设计。对于水平集方法，在这一领域鲜有文献报道。本文中，我们提出了一种基于 DNN 的水平集方法，以有效生成具有高结构性能的多样且有竞争力的设计。

本文的组织结构如下。第 2 节介绍基于深度神经网络的隐式建模。第 3 节详细描述 DNN 水平集拓扑优化公式。第 4 节给出数值算例以说明所提出的参数化水平集方法的有效性，最后第 5 节给出结论。

---

## 2. 深度神经网络隐式建模

为了用单一连续可微函数重建设计域，本文提出了一种基于深度神经网络的隐式建模方法。前馈网络 [49] 在输入层和输出层之间具有一个或多个层，主要用于函数逼近。DNN 的典型架构如图 1 所示，包含输入层、隐藏层和输出层。深度前馈神经网络的数学公式可以定义为：


$$\mathbb{N}(x, y, \boldsymbol{\theta}) = \mathbb{N}(\boldsymbol{a}^{(L+1)}(\boldsymbol{h}^{(L)}(\boldsymbol{a}^{(L)}(\dots \boldsymbol{h}^{(1)}(\boldsymbol{a}^{(1)}(x, y)))))) \tag{1}$$

其中 $\mathbb{N}$ 表示前馈网络，$\boldsymbol{\theta}$ 是网络的参数。隐藏层定义为 $\boldsymbol{h}^{(l)}(\boldsymbol{x})$，具有 $L$ 个隐藏层的网络可以表示为：

$$\boldsymbol{a}^{(l)}(\boldsymbol{x}) = \boldsymbol{W}^{(l)}\boldsymbol{x} + \boldsymbol{b}^{(l)} \tag{2}$$

其中 $\boldsymbol{W}^{(l)}$ 是权重矩阵，$\boldsymbol{b}^{(l)}$ 是第 $l$ 层的偏置向量。权重矩阵 $\boldsymbol{W}^{(l)}(l=1,2,\cdots L)$ 和偏置 $\boldsymbol{b}^{(l)}(l=1,2,\cdots L)$ 可以合并为一个参数 $\boldsymbol{\theta}$。$\boldsymbol{h}^{(l)}(l=1,2,\cdots L)$ 是隐藏层激活函数（核函数）。

> **图1. DNN 架构**：(a) 一个隐藏层 (b) 两个隐藏层 (c) 三个隐藏层

事实上，DNN 是非线性函数的通用逼近器。已经证明，三层前馈神经网络可以以任意精度逼近任何连续多元函数 [50]。DNN 是高维空间中函数逼近的非常有效的工具。除此之外，DNN 模型是解析可微的，基于图的计算自动微分 [51] 技术可以轻松获取梯度信息。与传统的离散水平集方法相比，DNN 的另一个显著优势是模型压缩。如文献所述，DeepSDF [45] 使用深度神经网络学习符号距离函数来表示复杂几何形状的零水平集，展示了非凡的模型压缩能力，其中 DeepSDF 所表示形状的高保真度得到了验证。

网络的初始权重对于训练的收敛至关重要。在实践中，通常将所有权重和偏置初始化为零均值的随机值 [52]。由 DNN 表示的隐式水平集函数的初始化可以公式化如下：

$$\begin{cases} \text{Find: } \boldsymbol{\theta} \\ \text{Min: } \sum_{i=1}^{N} \|\mathbb{N}(x_i, y_i, \boldsymbol{\theta}) - \Phi(x_i, y_i)\|^2 \end{cases} \tag{3}$$

其中 $\mathbb{N}$ 是前馈神经网络，$\Phi$ 是目标隐函数。算子 $\|\cdot\|^2$ 表示 2-范数。$(x, y)$ 表示点的坐标。这里应用反向传播学习算法 [52] 来训练神经网络。激活函数选择为双曲正切函数。每层网络包含 8 个神经元。目标隐函数如图 2 所示。使用三种不同架构对内部含有 5 个圆孔的板的训练结果如图 3 所示。注意，对于只有一个隐藏层的网络，训练后的形状无法达到高保真度，如图 3(a) 所示；而对于具有三个隐藏层的网络，可以获得更好的形状训练结果，如图 3(c) 所示。

> **图2. 目标隐函数**
> 
> **图3. 训练结果**：(a) 一个隐藏层 (b) 两个隐藏层 (c) 三个隐藏层

---

## 3. DNN 水平集方法用于结构拓扑优化

### 3.1 传统基于水平集的拓扑优化

传统水平集方法使用零等高线（二维）或等值面（三维）来表示几何边界，该方法由 Osher 和 Sethian [21] 引入，用于模拟动态界面的运动。界面由隐函数 $\Phi(\boldsymbol{x})$ 的零水平集描述，该函数在设计域内是 Lipschitz 连续的。在本文中，水平集函数 $\Phi(x)$ 定义为：

$$\begin{cases} \Phi(\boldsymbol{x}, t) > 0, & (\boldsymbol{x} \in \Omega) \\ \Phi(\boldsymbol{x}, t) = 0, & (\boldsymbol{x} \in \partial\Omega) \\ \Phi(\boldsymbol{x}, t) < 0, & (\boldsymbol{x} \in D \setminus \Omega) \end{cases} \tag{4}$$

其中 $D$ 是设计域，$\Omega$ 表示所有容许形状，$\partial\Omega$ 表示形状边界，$t$ 是形状动态演化的伪时间 [23]。通过对零水平集关于伪时间 $t$ 求导，可以得到 Hamilton-Jacobi PDE 如下：

$$\frac{\partial\Phi}{\partial t} - V_n|\nabla\Phi| = 0 \tag{5}$$

其中 $V_n$ 是通过灵敏度分析计算的法向速度。零水平集的形状通过求解上述 Hamilton-Jacobi 方程沿梯度方向演化。该方程可以使用迎风格式求解，其中需要重初始化过程作为辅助步骤，以避免隐函数在其零水平集附近变得过平或过陡。在本文中，目标选择为最小化结构柔度 $J(\Phi)$，可以公式化如下：

$$\min: J(\Phi) = \int_D (\boldsymbol{\varepsilon}(\boldsymbol{u}) : \boldsymbol{C} : \boldsymbol{\varepsilon}(\boldsymbol{u})) H(\Phi) d\Omega \tag{6}$$

约束条件为：

$$\begin{cases} \text{Volume} = \int_D H(\Phi) d\Omega = V_0 \\ a(\boldsymbol{u}, \boldsymbol{v}, \Phi) = l(\boldsymbol{v}, \Phi) \\ \boldsymbol{u} = \boldsymbol{u}_0 \quad \text{in } \Gamma_u \\ \boldsymbol{C} : \boldsymbol{\varepsilon}(\boldsymbol{u}) \cdot \boldsymbol{n} = \boldsymbol{\tau} \quad \text{in } \Gamma_{\tau} \end{cases} \tag{7}$$

其中上述方程中的符号可以写为：

$$a(\boldsymbol{u}, \boldsymbol{v}, \Phi) = \int_D (\boldsymbol{\varepsilon}(\boldsymbol{u}) : \boldsymbol{C} : \boldsymbol{\varepsilon}(\boldsymbol{v})) H(\Phi) d\Omega \tag{8}$$

$$l(\boldsymbol{v}, \Phi) = \int_{\Gamma_{\tau}} \boldsymbol{\tau} \cdot \boldsymbol{v} d\Gamma + \int_D \boldsymbol{b} \cdot \boldsymbol{v} H(\Phi) d\Omega \tag{9}$$

其中 $\boldsymbol{u}$ 是位移，$\boldsymbol{\varepsilon}(\boldsymbol{u})$ 是应变。$H(\cdot)$ 表示 Heaviside 阶跃函数。$\boldsymbol{C}$ 是弹性张量。$\Gamma_u$ 和 $\Gamma_{\tau}$ 分别表示位移边界和力边界。$\boldsymbol{u}_0$ 是给定的位移边界条件。$\boldsymbol{\tau}$ 和 $\boldsymbol{b}$ 分别表示边界上的牵引力和域内的体力。$a(\boldsymbol{u}, \boldsymbol{v}, \Phi)$ 是能量双线性形式，$l(\boldsymbol{v}, \Phi)$ 是载荷线性形式。$\boldsymbol{v}$ 是虚位移场。算子 $(:)$ 表示张量缩并。对于 Heaviside 阶跃函数，在孔洞区域等于零，在实体区域等于一。在实践中，阶跃 Heaviside 函数通过光滑函数近似，以确保在过渡区域可微。光滑 Heaviside 函数可以公式化如下：

$$H_{\delta}(\Phi) = \begin{cases} \delta & \Phi < -\Delta \\ 0.75(1 - \delta) \left(\frac{\Phi}{\Delta} - \frac{\Phi^3}{3\Delta^3}\right) + \frac{1 + \delta}{2} & -\Delta \leq \Phi \leq \Delta \\ 1 & \Phi > \Delta \end{cases} \tag{10}$$

其中 $\delta$ 是一个小值，$\Delta$ 表示过渡宽度的一半。光滑 Heaviside 函数的详细数学性质可参考文献 [24]。

### 3.2 DNN 水平集优化方法

本文提出了一种 DNN 水平集方法，将 Hamilton-Jacobi PDE 转化为设计域内的 ODE 系统用于拓扑优化。对于传统水平集方法，隐函数 $\Phi(x)$ 通过求解 Hamilton-Jacobi 方程来更新以获得最优拓扑。在本文中，DNN 隐式建模被应用于表示隐函数 $\Phi(x)$，其中 $\Phi(x)$ 的演化等价于更新网络的参数。如前一节所述，由时间依赖神经网络表示的隐函数可以表示为：

$$\Phi(\boldsymbol{x}, t) = \mathbb{N}(x, y, \boldsymbol{\theta}(t)) \tag{11}$$

将式 (11) 代入 Hamilton-Jacobi 方程，参数化 ODE 可以写为：

$$\frac{\partial\mathbb{N}(x, y, \boldsymbol{\theta}(t))}{\partial t} - V_n|\nabla(\mathbb{N}(x, y, \boldsymbol{\theta}(t)))| = 0 \tag{12}$$

$$\frac{\partial\mathbb{N}(x, y, \boldsymbol{\theta})}{\partial\boldsymbol{\theta}} \cdot \frac{\partial\boldsymbol{\theta}(t)}{\partial t} - V_n\left(\left(\frac{\partial(\mathbb{N}(x, y, \boldsymbol{\theta}(t)))}{\partial x}\right)^2 + \left(\frac{\partial(\mathbb{N}(x, y, \boldsymbol{\theta}(t)))}{\partial y}\right)^2\right)^{1/2} = 0 \tag{13}$$

应用矩阵 $\frac{\partial\mathbb{N}(x, y, \boldsymbol{\theta})}{\partial\boldsymbol{\theta}}$ 的 Moore-Penrose 伪逆 $\mathcal{M}^+$ 来获得上述系统的最小二乘解：

$$\frac{\partial\boldsymbol{\theta}(t)}{\partial t} - \mathcal{M}^+ V_n\left(\left(\frac{\partial(\mathbb{N}(x, y, \boldsymbol{\theta}(t)))}{\partial x}\right)^2 + \left(\frac{\partial(\mathbb{N}(x, y, \boldsymbol{\theta}(t)))}{\partial y}\right)^2\right)^{1/2} = 0 \tag{14}$$

其中 $\mathcal{M}^+$ 可以表示为：

$$\mathcal{M}^+ = \left(\left(\frac{\partial\mathbb{N}(x, y, \boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\right)^T \frac{\partial\mathbb{N}(x, y, \boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\right)^{-1} \left(\frac{\partial\mathbb{N}(x, y, \boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\right)^T \tag{15}$$

在式 (14) 中，DNN 系数是时间依赖的，其中 DNN 的初值可以通过 BP 训练获得。因此，一个 PDE 问题被转化为 ODE 问题。值得一提的是，DNN 关于其参数或输入的导数信息可以通过基于图的自动微分轻松获取。为了获得高精度和稳定的解，式 (14) 使用 Runge-Kutta-Fehlberg (RKF45) 方法 [53] 求解，该方法被文献 [54] 推荐。RKF45 方法能够确定是否使用了合适的步长，其中对解进行两次不同的近似并进行比较。如果两次近似在指定精度下不一致，则减小步长。该方法的更多细节可参考文献 [53]。一般而言，由于 Courant-Friedrichs-Lewy (CFL) 稳定性条件 [55]，时间步长应足够小以实现数值稳定性。

基于形状导数，最小柔度问题中沿自由移动边界的法向速度 $V_n$ 可以表示如下：

$$V_n = \boldsymbol{\varepsilon}(\boldsymbol{u}) : \boldsymbol{C} : \boldsymbol{\varepsilon}(\boldsymbol{u}) - \lambda \tag{16}$$

其中 $\lambda$ 是强制体积分数约束的 Lagrange 乘子。本文采用增广 Lagrange 更新方案 [56] 来更新 $\lambda$。

### 3.3 参数化重初始化方案

对于标准水平集方法，演化过程中可能出现不规则性，这不可避免地导致水平集演化的不稳定性。为克服这一困难，引入了重初始化方案来正则化水平集函数（LSF），以保持边界演化的稳定性。一般而言，重初始化通过周期性地将 LSF 重塑为符号距离函数来实现 [57]。标准的初始化方法是通过获得以下演化方程的稳态解：

$$\frac{\partial\Phi}{\partial t} = \text{sign}(\Phi_{\text{initial}})(1 - |\nabla\Phi|) \tag{18}$$

$$\Phi_{\text{initial}} = \Phi(t = 0)$$

其中 $\Phi_{\text{initial}}$ 是要初始化的水平集函数，$\text{sign}(\cdot)$ 表示符号函数。将式 (11) 代入式 (18) 中的重初始化方程，可以得到以下方程：

$$\frac{\partial\mathbb{N}(x, y, \theta)}{\partial\theta} \cdot \frac{\partial\theta}{\partial t} - \text{sign}(\Phi_{\text{initial}})(|\nabla\mathbb{N}(x, y, \theta)| - 1) = 0 \tag{19}$$

式 (19) 中使用的重初始化过程通常会轻微移动零水平集等高线，这可能在优化过程中导致不一致性。一般而言，重初始化过程需要周期性执行以保持符号距离函数 [13, 58]。Hartmann 等人 [59] 提出了约束重初始化方案，其中获得最小二乘解以保持零水平等高线的位置。Long 等人 [32] 提出了一种使用双阱势函数的新水平集方案，在拓扑优化循环内实现距离正则化。在本文中，周期性应用常规重初始化方案，以避免隐函数变得过平或过陡。更多实现细节可参考文献 [23]。式 (19) 可以使用 Runge-Kutta-Fehlberg (RKF45) 方法 [55] 求解。

所提出的基于 DNN 的水平集算法描述如下：

1. **初始化**参数化水平集函数 $\Phi(\boldsymbol{\theta})$，对应于初始猜测 $\Phi_0$。初始权重和偏置通过反向传播算法（式 (3)）确定。
2. **迭代直到收敛**，对于 $k \geq 0$：
    - (a) 基于隐函数 $\Phi_k$，通过 Heaviside 函数（式 (10)）计算设计域内的材料分布。使用 FEM 技术求解平衡方程（式 (7)）以获得位移场。
    - (b) 计算目标函数 $J(\Phi)$ 和法向速度 $V_n$。通过使用 RKF45 方法求解参数化 Hamilton-Jacobi 方程（式 (13)）来更新权重和偏置。
    - (c) 出于稳定性原因，每个迭代步通过求解式 (19) 对水平集函数 $\Phi$ 进行重初始化。

> **图4. 基于 DNN 的水平集方法流程图**

---

## 4. 数值算例

在本节中，给出了若干二维数值算例以验证所提出的 DNN 水平集方法的有效性。除非另有说明，选择以下参数：实体材料的弹性模量 $E = 1$，孔洞材料的模量 $E = 1 \times 10^{-6}$。泊松比选择为 $\nu = 0.3$。体积分数约束设置为 0.4。深度神经网络使用反向传播训练算法进行初始化。激活函数选择为双曲正切函数 [60]。详细的初始化描述见第 2 节。对于所有数值算例，设计域使用网格尺寸为 1 的矩形网格离散化。有限元分析基于"ersatz material"方法，这是一种公认的水平集拓扑优化方法 [23]。

### 4.1 MBB 梁

本算例研究 MBB 梁的最小柔度问题。边界条件如图 5 所示，其中集中力 $P = 1$ 施加在顶部边缘中点。网络架构如图 6 所示。注意，左下角为固定支撑，右下角为滚动支撑。设计域使用 $200 \times 100$ 个单元进行网格划分，网格尺寸为 1。固定 Lagrange 乘子选择为 $l = 5$，时间步长选择为 $\tau = 3 \times 10^{-3}$。对于第一种情况，隐函数由一个隐藏层的神经网络表示，每个隐藏层包含 8 个神经元（图 6）。设计变量总数为 33。为了生成对称设计，只使用设计域的一半进行优化，并施加对称边界条件。最终最优设计如图 8(a) 所示。初始设计（训练结果）如图 7(a) 所示。注意，具有一个隐藏层的神经网络是浅层网络，由于拟合能力有限，训练结果中的孔洞不是完美的圆。经过 150 次迭代后，通过求解式 (14) 中的 ODE 可以获得稳定的拓扑优化解（图 10(a)）。

> **图5. MBB 梁的柔度设计**
> 
> **图6. 网络架构**
> 
> **图7. MBB 梁设计的初始设计**（DNN 具有一个隐藏层）
> 
> **图8. MBB 梁的优化设计**
> 
> **图9. 优化设计的隐函数**
> 
> **图10. 收敛历史**：(a) 隐藏层：8 (b) 隐藏层：8×8 (c) 隐藏层：8×8×8
> 
> **图11. 基准设计**（柔度：33.48）

为了进行比较，选择具有两个隐藏层和三个隐藏层的神经网络来表示水平集的隐函数。优化参数设置与前述算例相同。网络架构如图 1(b) 和 (c) 所示。输入是设计域中的点坐标 $(x, y)$，输出是当前点的隐函数值。初始训练结果如图 7(b) 和 7(c) 所示，最优设计如图 8(b) 和 8(c) 所示。设计变量总数分别为 105（两个隐藏层）和 177（三个隐藏层）。最优设计的隐函数如图 9 所示。两个最优设计的柔度值分别为 32.57 和 39.42。收敛历史如图 10 所示。由于水平集函数通过更新神经网络参数来更新，从数学角度来看可以自由生成新的孔洞。这一显著特点在图 8 中得到了验证，其中优化过程中生成了新的小孔洞。使用 top88 代码 [61] 通过 SIMP 方法生成的基准设计如图 11 所示。注意，标准 SIMP 方法的滤波半径选择为 $r = 2$。SIMP 方法产生的最优设计柔度为 33.48。与基于 DNN 的水平集方法的解相比，优化设计（具有一层和两层网络）的结构柔度值与基准解非常接近。

为了进一步验证所提出的基于 DNN 的水平集方法在生成多样且具有竞争力的设计方面的有效性，选择了不同的网络架构来产生多样化的设计，如图 12 所示。值得一提的是，不同的网络能够生成明显不同的解。对于具有 5 个神经元的单层网络，优化结果简单，没有发现复杂的几何细节，如图 12(a) 所示。对于具有 2 层（15×15）的网络，优化设计似乎更加复杂，内部有若干类似桁架的支撑构件，如图 12(f) 所示。对于给定的神经网络（NN）架构，NN 所表示的隐函数（分割）是所有可能设计的一个子空间。因此，具有不同架构的 NN 描述了解的不同子空间，这解释了多样化解的原因。

> **图12. 基于 DNN 的水平集方法生成的多样且具有竞争力的设计**：
> (a) 柔度：31.52 (5) (b) 柔度：42.62 (10) (c) 柔度：38.45 (15)
> (d) 柔度：35.07 (5×5) (e) 柔度：40.23 (10×10) (f) 柔度：43.01 (15×15)
> (g) 柔度：32.41 (5×5×5) (h) 柔度：41.78 (10×10×10) (i) 柔度：44.32 (15×15×15)

### 4.2 短悬臂梁

短悬臂梁的最小柔度设计如图 13 所示。设计域是一个正方形，左侧为固定边界条件，右侧中点施加垂直集中力 $F = 1$。设计域使用 $100 \times 100$ 个单元进行网格划分，网格尺寸为 1。优化公式如式 (6) 和 (7) 所述。体积约束采用固定 Lagrange 乘子 $l = 3$，时间步长选择为 $\tau = 3 \times 10^{-3}$。对于第一种情况，选择具有一个隐藏层的神经网络来表示隐函数，其中隐藏层包含 8 个神经元。由于浅层神经网络的拟合能力有限，初始化形状在边界附近存在一些伪影，训练结果如图 14(a) 所示。最终最优设计如图 15(a) 所示。最优设计的隐函数如图 16(a) 所示，优化过程在 120 次迭代后收敛（图 17(a)）。为了与浅层神经网络生成的数值结果进行比较，这里检验了具有两个或三个隐藏层的网络。网络架构如图 1(b) 和图 1(c) 所示，每层包含 8 个神经元。设计变量总数分别为 105（两个隐藏层）和 177（三个隐藏层）。其他优化设置与之前相同。最优拓扑和隐函数如图 15(b-c) 和图 16(b-c) 所示。显然，使用深度神经网络（两个或三个隐藏层）的最优设计相比浅层神经网络获得的结果具有更复杂的几何特征。两种设计的收敛历史如图 17(b) 和 (c) 所示，其中稳定的拓扑优化设计在 140 次迭代后实现。使用 SIMP 方法获得的基准设计如图 18 所示。值得一提的是，具有 2 层或 3 层的 NN 产生的设计值略低于基准设计值（差异约 3.5%）。

> **图13. 短悬臂梁的柔度设计**
> 
> **图14. 短悬臂梁的初始设计**
> 
> **图15. 短悬臂梁的优化设计**：(a) 隐藏层：8 (181946) (b) 隐藏层：8×8 (176893) (c) 隐藏层：8×8×8 (177848)
> 
> **图16. 优化设计的隐函数**
> 
> **图17. 收敛历史**：(a) 隐藏层：8 (b) 隐藏层：8×8 (c) 隐藏层：8×8×8
> 
> **图18. 基准设计**（柔度：183543）

为了进一步生成多个备选方案，检验了 9 种不同的 NN 架构以获得解，如图 19 所示。显然，尽管这些设计具有明显不同的拓扑，但优化设计的柔度值相对于基准非常接近（差异小于 5%）。可以轻松获得非对称设计，如图 19(b)-(i) 所示。

> **图19. 基于 DNN 的水平集方法生成的多样且具有竞争力的设计**：
> (a) 柔度：174319 (5) (b) 柔度：176569 (10) (c) 柔度：175819 (15)
> (d) 柔度：197768 (5×5) (e) 柔度：174970 (10×10) (f) 柔度：181101 (15×15)
> (g) 柔度：190953 (5×5×5) (h) 柔度：176683 (10×10×10) (i) 柔度：178011 (15×15×15)

---

## 5. 结论

本文提出了一种用于拓扑优化的 DNN 水平集方法。深度神经网络在函数逼近方面非常流行。隐函数由深度前馈神经网络表示。激活函数选择为双曲正切函数。基于深度神经网络，可以实现隐函数梯度和曲率的高度光滑性。Hamilton-Jacobi PDE 被转化为参数化 ODE，隐函数通过更新网络的权重和偏置来更新。

所提出方法的主要贡献在于将 DNN 作为函数逼近器来描述水平集方法的隐函数。不同的 DNN 架构能够生成多样且具有竞争力的设计，具有优异的结构性能。基于 DNN 的水平集方法可以为设计者提供多个概念性备选方案，而不是仅仅寻找一个最大化或最小化目标的最优解。当前工作的局限性在于网络架构与结构复杂度或性能之间的数学联系无法显式量化。目前，应用数学工具来量化这种关系极其困难，未来将致力于解决这一问题。此外，使用深度学习方法来表示隐函数为机器学习与拓扑优化的结合开辟了机会。

---

## 致谢

感谢美国国家科学基金会（CMMI-1634261）对本工作的经费支持。

---

## 参考文献

[1] M. P. Bendsoe and O. Sigmund, *Topology optimization: theory, methods, and applications*. Springer Science & Business Media, 2013.

[2] O. Sigmund and K. Maute, "Topology optimization approaches," *Structural and Multidisciplinary Optimization*, vol. 48, no. 3, pp. 1031-1055, 2013.

[3] M. P. Bendsøe and O. Sigmund, "Material interpolation schemes in topology optimization," *Archive of applied mechanics*, vol. 69, no. 9-10, pp. 635-654, 1999.

[4] N. P. van Dijk, K. Maute, M. Langelaar, and F. Van Keulen, "Level-set methods for structural topology optimization: a review," *Structural and Multidisciplinary Optimization*, vol. 48, no. 3, pp. 437-472, 2013.

[5] F. Wang, B. S. Lazarov, and O. Sigmund, "On projection methods, convergence and robust formulations in topology optimization," *Structural and Multidisciplinary Optimization*, vol. 43, no. 6, pp. 767-784, 2011.

[6] J. Norato, B. Bell, and D. A. Tortorelli, "A geometry projection method for continuum-based topology optimization with discrete elements," *Computer Methods in Applied Mechanics and Engineering*, vol. 293, pp. 306-327, 2015.

[7] S. Watts and D. A. Tortorelli, "A geometric projection method for designing three‐dimensional open lattices with inverse homogenization," *International Journal for Numerical Methods in Engineering*, vol. 112, no. 11, pp. 1564-1588, 2017.

[8] B. S. Lazarov and F. Wang, "Maximum length scale in density based topology optimization," *Computer Methods in Applied Mechanics and Engineering*, vol. 318, pp. 826-844, 2017.

[9] M. Zhou, B. S. Lazarov, F. Wang, and O. Sigmund, "Minimum length scale in topology optimization by geometric constraints," *Computer Methods in Applied Mechanics and Engineering*, vol. 293, pp. 266-282, 2015.

[10] B. S. Lazarov, F. Wang, and O. Sigmund, "Length scale and manufacturability in density-based topology optimization," *Archive of Applied Mechanics*, vol. 86, no. 1-2, pp. 189-218, 2016.

[11] B. S. Lazarov, M. Schevenels, and O. Sigmund, "Robust design of large-displacement compliant mechanisms," *Mechanical sciences*, vol. 2, no. 2, pp. 175-182, 2011.

[12] J. K. Guest, "Topology optimization with multiple phase projection," *Computer Methods in Applied Mechanics and Engineering*, vol. 199, no. 1-4, pp. 123-135, 2009.

[13] J. K. Guest, "Imposing maximum length scale in topology optimization," *Structural and Multidisciplinary Optimization*, vol. 37, no. 5, pp. 463-473, 2009.

[14] J. K. Guest, J. H. Prévost, and T. Belytschko, "Achieving minimum length scale in topology optimization using nodal design variables and projection functions," *International journal for numerical methods in engineering*, vol. 61, no. 2, pp. 238-254, 2004.

[15] A. Asadpoure, M. Tootkaboni, and J. K. Guest, "Robust topology optimization of structures with uncertainties in stiffness–Application to truss structures," *Computers & Structures*, vol. 89, no. 11-12, pp. 1131-1141, 2011.

[16] M. Schevenels, B. S. Lazarov, and O. Sigmund, "Robust topology optimization accounting for spatially varying manufacturing errors," *Computer Methods in Applied Mechanics and Engineering*, vol. 200, no. 49-52, pp. 3613-3627, 2011.

[17] O. Sigmund, "Manufacturing tolerant topology optimization," *Acta Mechanica Sinica*, vol. 25, no. 2, pp. 227-239, 2009.

[18] B. S. Lazarov, M. Schevenels, and O. Sigmund, "Topology optimization with geometric uncertainties by perturbation techniques," *International Journal for Numerical Methods in Engineering*, vol. 90, no. 11, pp. 1321-1336, 2012.

[19] O. Sigmund, "Morphology-based black and white filters for topology optimization," *Structural and Multidisciplinary Optimization*, vol. 33, no. 4-5, pp. 401-424, 2007.

[20] J. A. Sethian, "Theory, algorithms, and applications of level set methods for propagating interfaces," *Acta numerica*, vol. 5, pp. 309-395, 1996.

[21] S. Osher and J. A. Sethian, "Fronts propagating with curvature-dependent speed: algorithms based on Hamilton-Jacobi formulations," *Journal of computational physics*, vol. 79, no. 1, pp. 12-49, 1988.

[22] S. J. Osher and F. Santosa, "Level set methods for optimization problems involving geometry and constraints: I. Frequencies of a two-density inhomogeneous drum," *Journal of Computational Physics*, vol. 171, no. 1, pp. 272-288, 2001.

[23] G. Allaire, F. Jouve, and A.-M. Toader, "Structural optimization using sensitivity analysis and a level-set method," *Journal of computational physics*, vol. 194, no. 1, pp. 363-393, 2004.

[24] M. Y. Wang, X. Wang, and D. Guo, "A level set method for structural topology optimization," *Computer methods in applied mechanics and engineering*, vol. 192, no. 1-2, pp. 227-246, 2003.

[25] S. Wang and M. Y. Wang, "Radial basis functions and level set method for structural topology optimization," *International journal for numerical methods in engineering*, vol. 65, no. 12, pp. 2060-2090, 2006.

[26] P. Wei and M. Y. Wang, "Piecewise constant level set method for structural topology optimization," *International Journal for Numerical Methods in Engineering*, vol. 78, no. 4, pp. 379-402, 2009.

[27] L. Jiang, S. Chen, and X. Jiao, "Parametric shape and topology optimization: A new level set approach based on cardinal basis functions," *International Journal for Numerical Methods in Engineering*, vol. 114, no. 1, pp. 66-87, 2018.

[28] X. Guo, W. Zhang, and W. Zhong, "Doing topology optimization explicitly and geometrically—a new moving morphable components based framework," *Journal of Applied Mechanics*, vol. 81, no. 8, 2014.

[29] W. Zhang, J. Zhou, Y. Zhu, and X. Guo, "Structural complexity control in topology optimization via moving morphable component (MMC) approach," *Structural and Multidisciplinary Optimization*, vol. 56, no. 3, pp. 535-552, 2017.

[30] W. Zhang et al., "Explicit three dimensional topology optimization via Moving Morphable Void (MMV) approach," *Computer Methods in Applied Mechanics and Engineering*, vol. 322, pp. 590-614, 2017.

[31] W. Zhang, D. Li, J. Zhou, Z. Du, B. Li, and X. Guo, "A moving morphable void (MMV)-based explicit approach for topology optimization considering stress constraints," *Computer Methods in Applied Mechanics and Engineering*, vol. 334, pp. 381-413, 2018.

[32] L. Jiang and S. Chen, "Parametric structural shape & topology optimization with a variational distance-regularized level set method," *Computer Methods in Applied Mechanics and Engineering*, vol. 321, pp. 316-336, 2017.

[33] Y. Luo, J. Xing, and Z. Kang, "Topology optimization using material-field series expansion and Kriging-based algorithm: An effective non-gradient method," *Computer Methods in Applied Mechanics and Engineering*, vol. 364, p. 112966, 2020.

[34] P. Lison, "An introduction to machine learning," Language Technology Group: Edinburgh, UK, 2015.

[35] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, "Xnor-net: Imagenet classification using binary convolutional neural networks," in *European conference on computer vision*, 2016: Springer, pp. 525-542.

[36] E. Gawehn, J. A. Hiss, and G. Schneider, "Deep learning in drug discovery," *Molecular informatics*, vol. 35, no. 1, pp. 3-14, 2016.

[37] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, vol. 378, pp. 686-707, 2019.

[38] M. Raissi, A. Yazdani, and G. E. Karniadakis, "Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations," *Science*, vol. 367, no. 6481, pp. 1026-1030, 2020.

[39] R. Iten, T. Metger, H. Wilming, L. Del Rio, and R. Renner, "Discovering physical concepts with neural networks," *Physical Review Letters*, vol. 124, no. 1, p. 010508, 2020.

[40] S. L. Brunton, B. R. Noack, and P. Koumoutsakos, "Machine learning for fluid mechanics," *Annual Review of Fluid Mechanics*, vol. 52, pp. 477-508, 2020.

[41] M. Raissi, Z. Wang, M. S. Triantafyllou, and G. E. Karniadakis, "Deep learning of vortex-induced vibrations," *Journal of Fluid Mechanics*, vol. 861, pp. 119-137, 2019.

[42] Y. Yu, T. Hur, J. Jung, and I. G. Jang, "Deep learning for determining a near-optimal topological design without any iteration," *Structural and Multidisciplinary Optimization*, vol. 59, no. 3, pp. 787-799, 2019.

[43] X. Lei, C. Liu, Z. Du, W. Zhang, and X. Guo, "Machine learning-driven real-time topology optimization under moving morphable component-based framework," *Journal of Applied Mechanics*, vol. 86, no. 1, 2019.

[44] S. Oh, Y. Jung, S. Kim, I. Lee, and N. Kang, "Deep generative design: Integration of topology optimization and generative models," *Journal of Mechanical Design*, vol. 141, no. 11, 2019.

[45] J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, "Deepsdf: Learning continuous signed distance functions for shape representation," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2019, pp. 165-174.

[46] B. Wang, Y. Zhou, Y. Zhou, S. Xu, and B. Niu, "Diverse competitive design for topology optimization," *Structural and Multidisciplinary Optimization*, vol. 57, no. 2, pp. 891-902, 2018.

[47] K. Yang et al., "Simple and effective strategies for achieving diverse and competitive structural designs," *Extreme Mechanics Letters*, vol. 30, p. 100481, 2019.

[48] Y. He, K. Cai, Z.-L. Zhao, and Y. M. Xie, "Stochastic approaches to generating diverse and competitive structural designs in topology optimization," *Finite Elements in Analysis and Design*, vol. 173, p. 103399, 2020.

[49] I. Goodfellow, Y. Bengio, and A. Courville, *Deep learning*. MIT press, 2016.

[50] G. Cybenko, "Approximation by superpositions of a sigmoidal function," *Mathematics of control, signals and systems*, vol. 2, no. 4, pp. 303-314, 1989.

[51] A. Paszke et al., "Automatic differentiation in pytorch," 2017.

[52] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning internal representations by error propagation," California Univ San Diego La Jolla Inst for Cognitive Science, 1985.

[53] J. C. Butcher, *The numerical analysis of ordinary differential equations: Runge-Kutta and general linear methods*. Wiley-Interscience, 1987.

[54] S. Wang, K. M. Lim, B. C. Khoo, and M. Y. Wang, "An extended level set method for shape and topology optimization," *Journal of Computational Physics*, vol. 221, no. 1, pp. 395-421, 2007.

[55] S. Osher, R. Fedkiw, and K. Piechor, "Level set methods and dynamic implicit surfaces," *Appl. Mech. Rev.*, vol. 57, no. 3, pp. B15-B15, 2004.

[56] M. Y. Wang and P. Wang, "The augmented Lagrangian method in structural shape and topology optimization with RBF based level set method," in *CJK-OSM 4: The Fourth China-Japan-Korea Joint Symposium on Optimization of Structural and Mechanical Systems*, 2006, p. 191.

[57] C. Li, C. Xu, C. Gui, and M. D. Fox, "Distance regularized level set evolution and its application to image segmentation," *IEEE transactions on image processing*, vol. 19, no. 12, pp. 3243-3254, 2010.

[58] V. J. Challis, "A discrete level-set topology optimization code written in Matlab," *Structural and multidisciplinary optimization*, vol. 41, no. 3, pp. 453-464, 2010.

[59] D. Hartmann, M. Meinke, and W. Schröder, "The constrained reinitialization equation for level set methods," *Journal of computational physics*, vol. 229, no. 5, pp. 1514-1535, 2010.

[60] G. A. Anastassiou, "Multivariate hyperbolic tangent neural network approximation," *Computers & Mathematics with Applications*, vol. 61, no. 4, pp. 809-821, 2011.

[61] E. Andreassen, A. Clausen, M. Schevenels, B. S. Lazarov, and O. Sigmund, "Efficient topology optimization in MATLAB using 88 lines of code," *Structural and Multidisciplinary Optimization*, vol. 43, no. 1, pp. 1-16, 2011.

---

> **说明**：本文档是对论文 *A Parametric Level Set Method for Topology Optimization based on Deep Neural Network (DNN)*（arXiv:2101.03286v1, 2021）的完整中文翻译。原文中的图表编号和公式编号均保留，但由于翻译为纯文本格式，图表本身未包含在内，请参阅原始 PDF 获取完整图表。