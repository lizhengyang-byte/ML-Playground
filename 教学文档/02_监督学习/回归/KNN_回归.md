# KNN 回归：用邻居的均值做预测

> 所属模块：02_监督学习/回归 | 源文件：KNN_回归.py | 核心功能：K 值选择、距离加权、距离度量对比

## 概述

KNN 回归是 KNN 分类的回归版本——找到 K 个最近邻，取它们目标值的平均（或加权平均）作为预测。没有训练过程，所有计算都在预测时完成。

脚本对比了不同 K 值、权重策略和距离度量对回归性能的影响。

## 关键代码解释

### 距离加权

`python
KNeighborsRegressor(n_neighbors=5, weights="distance")
`

uniform：简单平均。distance：权重 = 1/distance，近邻影响更大。distance 加权通常在 K 较大时表现更好。

## 使用示例

`python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=5, weights="distance"))])
pipe.fit(X_train, y_train)
`

## 注意事项

1. **必须特征缩放**
2. **预测速度慢**：O(n·d) 每个预测样本
3. **高维失效**：维度灾难使距离变得无意义
4. **不学习显式模型**：无法获得特征重要性或系数解读

## 延伸思考

- **RadiusNeighborsRegressor**：用固定半径而非固定 K 选择邻居
- **局部加权回归（LWR）**：KNN 的平滑版本，用核函数给距离赋权
- **FAISS 等 ANN 库**：加速大规模最近邻搜索
﻿## 数学原理

### 1. KNN 回归的预测公式

**代码对应**：`KNeighborsRegressor(n_neighbors=k, weights="uniform")` 和 `weights="distance"`。

对于查询样本 $\mathbf{x}$，找到 $K$ 个最近邻 $\mathcal{N}_K(\mathbf{x}) = \{i_1, i_2, \ldots, i_K\}$，预测值为：

**等权平均**（`weights="uniform"`）：

$$\hat{y}(\mathbf{x}) = \frac{1}{K}\sum_{i \in \mathcal{N}_K(\mathbf{x})} y_i$$

**距离加权平均**（`weights="distance"`）：

$$\hat{y}(\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_K(\mathbf{x})} w_i y_i}{\sum_{i \in \mathcal{N}_K(\mathbf{x})} w_i}, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$$

**代码对应**：代码中对比了 `uniform` 和 `distance` 两种权重策略。距离加权让更近的邻居有更大影响，通常效果更好。

### 2. 距离度量

**代码对应**：`metric="minkowski", p=p` 定义了 Minkowski 距离族。

闵可夫斯基距离：

$$d_p(\mathbf{x}, \mathbf{z}) = \left(\sum_{j=1}^{p} |x_j - z_j|^p\right)^{1/p}$$

- $p = 1$：曼哈顿距离（L1）
- $p = 2$：欧氏距离（L2，最常用）
- $p \to \infty$：切比雪夫距离

### 3. K 的偏差-方差权衡

**代码对应**：`for k in [1, 3, 5, ..., 50]` 展示了 K 对训练/测试 R² 的影响。

- $K = 1$：预测值等于最近邻的 $y$ 值。训练集上完美拟合（$R^2 = 1$），但泛化极差（高方差）
- $K = N$：预测值始终为全局均值 $\bar{y}$（高偏差，零方差）
- 最优 $K$ 在两者之间，通过交叉验证选择

**数学分析**：KNN 回归的期望误差可以分解为：

$$\mathbb{E}[(y - \hat{y})^2] = \underbrace{\left(\mathbb{E}[\hat{y}] - f(\mathbf{x})\right)^2}_{\text{Bias}^2} + \underbrace{\text{Var}[\hat{y}]}_{\text{Variance}} + \sigma^2$$

- $K$ 增大 $\Rightarrow$ 偏差增大（邻居平均值趋近全局均值），方差减小
- $K$ 减小 $\Rightarrow$ 偏差减小，方差增大

### 4. 维度灾难

KNN 在高维空间中失效。设数据均匀分布在 $[0,1]^p$ 的单位超立方体中，$K$ 个最近邻到查询点的平均距离为：

$$d_K \sim \left(\frac{K}{N}\right)^{1/p}$$

当 $p$ 很大时，即使 $N = 10000$，$K = 5$，最近邻的平均距离也接近 1——即最近邻和最远邻"一样远"。此时 KNN 的预测质量急剧下降。

**实际建议**：$p > 20$ 时 KNN 效果开始明显下降，考虑降维或使用其他模型。

### 5. 训练与预测复杂度

| 阶段 | 朴素实现 | KD-Tree（$p < 20$） | Ball Tree |
|------|---------|---------------------|-----------|
| 训练 | $O(1)$（存储即可） | $O(np\log n)$ | $O(np\log n)$ |
| 预测（每个样本） | $O(np)$ | $O(p\log n)$ | $O(p\log n)$ |

KNN 是**懒惰学习**（lazy learning）的典型代表：训练阶段几乎不做计算，所有计算推迟到预测阶段。这意味着预测速度慢，不适合实时应用。
