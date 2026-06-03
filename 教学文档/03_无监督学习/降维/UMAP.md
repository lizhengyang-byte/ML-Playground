# UMAP：比 t-SNE 更快更好的非线性降维

> 所属模块：03_无监督学习/降维 | 源文件：UMAP.py | 核心功能：非线性降维、n_neighbors/min_dist 调参、支持 transform 新数据

## 概述

UMAP（Uniform Manifold Approximation and Projection）是 2018 年提出的非线性降维算法，迅速成为 t-SNE 的主流替代品。它的三大优势：**更快**（适合大数据集）、**更好地保持全局结构**、**支持 transform 新数据**。

理论基础来自黎曼几何和代数拓扑，但实际使用起来和 t-SNE 一样简单。

脚本在 Iris 数据集上对比了 UMAP、t-SNE 和 PCA 的效果，并展示了 n_neighbors、min_dist 和 metric 参数的影响。

## 关键代码解释

### 两个核心参数

`python
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
`

- **n_neighbors**：邻域大小。越大越关注全局结构（大尺度模式），越小越关注局部细节。默认 15 通常效果不错。
- **min_dist**：嵌入空间中点之间的最小距离。越小簇越紧凑（容易看出簇的边界），越大簇越松散。

### 支持多种距离度量

`python
for metric in ["euclidean", "cosine", "manhattan", "chebyshev"]:
`

UMAP 支持丰富的距离函数。cosine 适合文本向量，euclidean 适合数值数据。

## 使用示例

`python
import umap
X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
`

## 注意事项

1. **需要安装 umap-learn**：pip install umap-learn
2. **结果有随机性**：设置 random_state 保证可复现
3. **n_neighbors 调参**：默认 15，大数据集可以增大到 50-100
4. **可以 transform 新数据**：
educer.transform(X_new)
5. **理论争议**：部分学者质疑其数学基础，但实际效果广泛认可

## 延伸思考

- **监督 UMAP**：利用标签信息指导降维，同类更紧凑
- **半监督 UMAP**：只用部分标签
- **UMAP 作为特征工程**：降维后接分类器，效果有时比原始特征更好
- **Parametric UMAP**：用神经网络学习 UMAP 映射，支持快速 transform
﻿## 数学原理

### 1. UMAP 的理论基础：黎曼几何与代数拓扑

**代码对应**：`umap.UMAP(n_neighbors=15, min_dist=0.1)` 训练 UMAP。

UMAP（Uniform Manifold Approximation and Projection）基于三个假设：
1. 数据均匀分布在黎曼流形上
2. 黎曼度量是局部恒定的（或近似恒定）
3. 流形是局部连通的

### 2. 高维模糊拓扑构建

**步骤 1**：对每个样本 $\mathbf{x}_i$，找到 $k$ 个最近邻（`n_neighbors` 参数）。

**步骤 2**：构建加权图，边权重为：

$$w_{ij} = \exp\left(-\frac{d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i}{\sigma_i}\right)$$

其中 $\rho_i$ 为 $\mathbf{x}_i$ 到其最近邻的距离（局部归一化），$\sigma_i$ 由 $k$ 近邻的约束确定。

**步骤 3**：对称化（模糊联合）：

$$p_{ij} = w_{ij} + w_{ji} - w_{ij} \cdot w_{ji}$$

这构建了高维数据的**模糊单纯复形**（fuzzy simplicial complex）。

### 3. 低维嵌入优化

在低维空间中，使用**逻辑曲线**（logistic curve）作为核函数：

$$q_{ij} = \frac{1}{1 + a\|\mathbf{y}_i - \mathbf{y}_j\|^{2b}}$$

其中 $a$ 和 $b` 由 `min_dist` 参数确定。

优化目标为交叉熵：

$$\text{CE} = \sum_{i \neq j}\left[p_{ij}\log\frac{p_{ij}}{q_{ij}} + (1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}\right]$$

与 t-SNE 只有 KL 散度（$p\log(p/q)$）不同，UMAP 的交叉熵还包含 $(1-p)\log((1-p)/(1-q))$ 项——这使得 UMAP 能更好地保持**全局结构**。

### 4. UMAP vs t-SNE

| 特性 | t-SNE | UMAP |
|------|-------|------|
| 理论基础 | 概率分布匹配 | 黎曼几何 + 代数拓扑 |
| 损失函数 | KL 散度 | 交叉熵 |
| 全局结构 | 差 | 较好 |
| 计算速度 | 慢 $O(n^2)$ | 快 $O(n^{1.14})$ |
| 可复现性 | 不稳定 | 较稳定 |
| 新数据投影 | 不支持 | 支持 `transform` |

### 5. 关键参数

- `n_neighbors`：局部邻域大小。大值保持全局结构，小值保持局部细节
- `min_dist`：低维空间中点的最小距离。小值使嵌入更紧凑，大值使嵌入更松散
- `n_components`：目标维度（通常 2 或 3）
- `metric`：距离度量（euclidean、cosine、manhattan 等）
