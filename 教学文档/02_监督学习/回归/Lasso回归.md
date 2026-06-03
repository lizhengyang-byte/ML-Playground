# Lasso 回归：L1 正则化的自动特征选择

> 所属模块：02_监督学习/回归 | 源文件：Lasso回归.py | 核心功能：L1 正则化特征选择、LassoCV 自动选参

## 概述

Lasso（Least Absolute Shrinkage and Selection Operator）回归在损失函数中加入 L1 惩罚项 α||w||₁。与岭回归的关键区别：L1 正则化会把部分系数**精确压缩为 0**，从而实现自动特征选择。

在高维数据中（特征远多于样本，或大部分特征是噪声），Lasso 能自动"筛"出重要特征，这是岭回归做不到的。

脚本构造了一个只有 3 个有用特征的稀疏数据集，演示 Lasso 如何自动识别这些特征，以及 alpha 对稀疏性的影响。

## 关键代码解释

### 特征选择效果

`python
true_coef = np.zeros(p)
true_coef[0] = 5.0; true_coef[3] = -3.0; true_coef[7] = 2.0
# ...
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
np.count_nonzero(lasso.coef_)  # 大约 3-5 个非零系数
`

Lasso 准确地把不重要特征的系数压到了 0，保留了 3 个真正有用的特征。这就是 L1 正则化的"稀疏诱导"特性。

### LassoCV 自动选参

`python
lasso_cv = LassoCV(alphas=None, cv=5, max_iter=10000, random_state=42)
`

lphas=None 让 LassoCV 自动生成一个 alpha 网格，用交叉验证选择最优值。

## 使用示例

`python
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
print(f"最优 alpha: {lasso_cv.alpha_}")
print(f"非零系数: {np.count_nonzero(lasso_cv.coef_)}")
`

## 注意事项

1. **必须特征缩放**：L1 正则化对系数尺度敏感
2. **最多选 min(n, p) 个特征**：Lasso 的选择数量受限于样本数
3. **共线性特征组**：Lasso 倾向于从一组高度相关的特征中只选一个，选择不稳定
4. **max_iter**：Lasso 使用坐标下降法，可能需要更多迭代才能收敛
5. **alpha 越大，非零系数越少**：需要在稀疏性和拟合能力之间找平衡

## 延伸思考

- **Lasso 的几何直觉**：L1 约束区域是菱形，最优解容易落在角点（即某些坐标为 0）
- **Basis Pursuit**：信号处理中的 L1 最小化，与 Lasso 思想相同
- **ElasticNet**：结合 L1 和 L2，解决 Lasso 在共线性组中的选择不稳定性
- **贝叶斯 Lasso**：用拉普拉斯先验替代 L1 罚项，提供不确定性估计
﻿## 数学原理

### 1. Lasso 的目标函数

**代码对应**：`Lasso(alpha=0.1).fit(X_train, y_train)` 使用 L1 正则化。

Lasso（Least Absolute Shrinkage and Selection Operator）的目标函数为：

$$J(\mathbf{w}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha\|\mathbf{w}\|_1 = \frac{1}{2n}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \alpha\sum_{j=1}^{p}|w_j|$$

与岭回归（L2 惩罚）的关键区别：L1 范数 $\|\mathbf{w}\|_1 = \sum|w_j|$ 在零点不可导，导致最优解中部分系数**精确等于零**。

### 2. L1 为何能产生稀疏解——几何直觉

在二维空间中，L1 约束 $\|w\|_1 \leq t$ 的可行域是**菱形**（diamond），L2 约束 $\|w\|_2 \leq t$ 的可行域是**圆形**。

损失函数的等高线是椭圆。当椭圆与菱形相切时，切点极大概率在**顶点**（即某个坐标轴上），此时对应的 $w_j = 0$。而椭圆与圆形相切时，切点通常不在坐标轴上。

这就是 L1 产生稀疏解（特征选择）而 L2 不会的几何本质。

### 3. L1 的最优性条件

L1 正则化问题没有闭式解，需要用坐标下降（Coordinate Descent）或近端梯度下降（Proximal Gradient Descent）求解。

对单个系数 $w_j$ 的软阈值（Soft Thresholding）更新：

$$w_j \leftarrow S_{\alpha/c}\left(\frac{1}{n}\sum_{i=1}^n x_{ij}(y_i - \hat{y}_i^{(-j)})\right)$$

其中软阈值函数为：

$$S_\lambda(z) = \text{sign}(z) \max(|z| - \lambda, 0) = \begin{cases} z - \lambda & z > \lambda \\ 0 & |z| \leq \lambda \\ z + \lambda & z < -\lambda \end{cases}$$

当 $|z| \leq \lambda$ 时，$w_j$ 被精确置零——这就是特征选择的数学机制。

**代码对应**：sklearn 的 `Lasso` 内部使用坐标下降法。`max_iter=10000` 控制最大迭代次数。

### 4. 稀疏性与 alpha 的关系

**代码对应**：代码中 `for alpha in [0.001, 0.01, ..., 5.0]` 展示了 alpha 对非零系数数的影响。

alpha 越大，软阈值 $\lambda = \alpha/c$ 越大，被置零的系数越多：

- $\alpha = 0$：无正则化，等价于 OLS，所有系数非零
- $\alpha \to \infty$：所有系数为 0
- 最优 $\alpha$ 通过交叉验证选择：`LassoCV(alphas=None, cv=5)`

### 5. Lasso 的局限性

**Lasso 最多选择 $\min(n, p)$ 个特征**。当 $p > n$ 时，Lasso 最多选出 $n$ 个非零系数。

**相关特征问题**：当多个特征高度相关时，Lasso 倾向于只选择其中一个（任意选择），而将其他相关特征的系数置零。这可能导致模型不稳定。

**数学原因**：L1 菱形的顶点只有 $2p$ 个，每个顶点只有一个非零坐标。当多个相关特征的"贡献"相近时，菱形顶点附近的最优解可能在不同特征之间跳跃。

### 6. Lasso 的贝叶斯解释

Lasso 等价于对权重施加**拉普拉斯先验**（Laplace prior）：

$$P(w_j) = \frac{\alpha}{2}\exp(-\alpha|w_j|)$$

拉普拉斯分布在零点有尖峰，因此 MAP 估计倾向于产生零系数。对比：

| 正则化 | 先验分布 | 零点密度 | 稀疏性 |
|--------|---------|---------|--------|
| L1 (Lasso) | 拉普拉斯 $\text{Lap}(0, 1/\alpha)$ | 高（尖峰） | 产生稀疏解 |
| L2 (Ridge) | 高斯 $\mathcal{N}(0, 1/\alpha)$ | 低（平滑） | 不产生稀疏解 |
