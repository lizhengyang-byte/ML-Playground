# t-SNE：高维数据的"显微镜"

> 所属模块：03_无监督学习/降维 | 源文件：t_SNE.py | 核心功能：非线性降维可视化、perplexity 调参、与 PCA 对比

## 概述

t-SNE（t-distributed Stochastic Neighbor Embedding）是目前最流行的高维数据可视化工具。它把高维数据映射到 2D 或 3D 空间，同时**保持数据的局部邻域结构**——高维空间中靠近的点在低维空间中依然靠近。

核心思想：在高维空间中用高斯分布建模点对的相似性，在低维空间中用 t 分布建模，然后最小化两个分布之间的 KL 散度。

脚本在 Iris 和 Digits 数据集上演示了 t-SNE 的降维效果和参数影响。

## 关键代码解释

### perplexity 参数

`python
for perp in [5, 10, 20, 30, 50]:
    tsne_p = TSNE(n_components=2, perplexity=perp, init="pca", random_state=42)
`

perplexity 可以粗略理解为"每个点关注多少个邻居"。通常 5-50，数据集大时用大值。太小 → 所有点聚成一团；太大 → 簇被挤在一起。

### PCA 预降维加速

`python
pca = PCA(n_components=30)
X_digits_pca = pca.fit_transform(X_digits)
tsne_d = TSNE(n_components=2).fit_transform(X_digits_pca)
`

t-SNE 的计算复杂度是 O(n²)，直接对 64 维数据做很慢。先用 PCA 降到 30 维可以大幅加速，同时去除噪声。

## 使用示例

`python
from sklearn.manifold import TSNE
X_2d = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)
`

## 注意事项

1. **只用于可视化**：不建议用于特征工程（不可逆、不可复现、不保持全局结构）
2. **每次运行结果不同**：除非固定 random_state
3. **不保持全局结构**：只保持局部邻域关系，簇间距离没有意义
4. **大数据集慢**：O(n²)，先用 PCA 预降维
5. **不能 transform 新数据**：每次需要重新运行整个算法

## 延伸思考

- **openTSNE**：更快的 t-SNE 实现，支持 transform 新数据
- **FIt-SNE**：基于 FFT 加速的 t-SNE，适合百万级数据
- **t-SNE 的 perplexity 与信息论的关系**：perplexity = 2^H，H 是条件熵
﻿## 数学原理

### 1. 高维空间中的概率分布

**代码对应**：`TSNE(n_components=2, perplexity=30)` 执行 t-SNE 降维。

t-SNE 在高维空间中定义样本对之间的**条件概率**：

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

对称化后：

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

$\sigma_i$ 由 **perplexity** 参数确定：

$$\text{Perp}(P_i) = 2^{H(P_i)}, \quad H(P_i) = -\sum_{j}p_{j|i}\log_2 p_{j|i}$$

perplexity 可以理解为有效邻居数。**代码对应**：`perplexity=30` 意味着每个点大约有 30 个"有效邻居"。

### 2. 低维空间中的概率分布

在低维空间（通常 2D）中，使用**Student-t 分布**（自由度为 1，即柯西分布）：

$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

**为什么用 t 分布而非高斯？** t 分布的尾部更重，能缓解"拥挤问题"（crowding problem）——高维空间中等距的点在低维空间中无法全部保持等距。

### 3. KL 散度损失函数

t-SNE 最小化高维分布 $P$ 和低维分布 $Q$ 之间的 KL 散度：

$$\text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log\frac{p_{ij}}{q_{ij}}$$

**代码对应**：`tsne.kl_divergence_` 返回最终的 KL 散度值。

KL 散度是**非对称**的：
- $p_{ij}$ 大、$q_{ij}$ 小时（高维近但低维远），惩罚重 → **保持局部结构**
- $p_{ij}$ 小、$q_{ij}$ 大时（高维远但低维近），惩罚轻 → **允许全局变形**

这就是 t-SNE 擅长保持局部结构但不保持全局距离的原因。

### 4. 梯度下降优化

对低维坐标 $\mathbf{y}_i$ 的梯度为：

$$\frac{\partial\text{KL}}{\partial\mathbf{y}_i} = 4\sum_{j}(p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$

直觉：
- $p_{ij} > q_{ij}$（高维近但低维不够近）：吸引 $\mathbf{y}_i$ 向 $\mathbf{y}_j$ 移动
- $p_{ij} < q_{ij}$（高维远但低维太近）：排斥 $\mathbf{y}_i$ 远离 $\mathbf{y}_j$

### 5. PCA 初始化

**代码对应**：`init="pca"` 用 PCA 的投影结果作为 t-SNE 的初始坐标。

PCA 初始化比随机初始化的优势：
- 收敛更快（初始位置更接近最优解）
- 结果更稳定（减少随机性）
- 避免差的局部最优

### 6. 计算复杂度与加速

朴素 t-SNE 需要 $O(n^2)$ 计算所有点对的距离。加速方法：
- **Barnes-Hut 近似**：$O(n\log n)$，用四叉树/八叉树远距离点对近似为一个点
- **先用 PCA 预降维**：将高维数据降到 30-50 维后再做 t-SNE

**代码对应**：代码中先用 PCA 降到 30 维再做 t-SNE，大幅加速。
