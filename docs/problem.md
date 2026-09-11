问题，弹性特征值方程

$$
Ae(u) = \omega^2 \rho u 
$$

边界条件

$$
u = 0 ，in\quad left\quad and\quad right. 
$$

$$
\sigma \cdot n = 0, in\quad top\quad and\quad bottom
$$

$$
\sigma \cdot n = 0 , in\quad boundary\quad of\quad hole\quad and\quad material
$$

更新$\rho$

$$
\min \omega^2 + \int S dx - V
$$

$$
\omega^2 = Rayleigh, S = H(SDF), \rho = \hat{\rho}S
$$

以上是问题，要求弹性特征值方程在满足面积约束下，最小特征值最大时的最有拓扑，其中$\omega^2$是特征值，$\rho$是密度，SDF是距离函数，H是heaviside变化，即SDF大于0代表有材料，S=1，SDF小于0代表空洞，S = 0。$\hat{\rho}$是密度分布，例如中心有个重块的密度是100，其他材料密度是1，空洞密度是0.

传统方法通过对min这一项进行灵敏性分析，然后通过交替迭代的方法对$\rho$进行求解($\rho$初始全1)，即先通过FEM求解弹性特征值问题得到最小特征值的特征对，然后再根据灵敏性分析更新$\rho$。

现在我要用神经网络替代传统方法， 分别构造5个网络代表应变应力，再构造第6个网络代表SDF，然后用交替的方法，在更新$\rho$上希望借鉴HJB方程的方案，目前我还不是很懂这个方案，需要深入了解一下
