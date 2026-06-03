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