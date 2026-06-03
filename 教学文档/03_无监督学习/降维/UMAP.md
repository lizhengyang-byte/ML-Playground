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
4. **可以 transform 新数据**：educer.transform(X_new)
5. **理论争议**：部分学者质疑其数学基础，但实际效果广泛认可

## 延伸思考

- **监督 UMAP**：利用标签信息指导降维，同类更紧凑
- **半监督 UMAP**：只用部分标签
- **UMAP 作为特征工程**：降维后接分类器，效果有时比原始特征更好
- **Parametric UMAP**：用神经网络学习 UMAP 映射，支持快速 transform