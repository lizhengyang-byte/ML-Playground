# KMeans 聚类：最简单也最常用的聚类算法

> 所属模块：03_无监督学习/聚类 | 源文件：KMeans.py | 核心功能：KMeans 聚类、肘部法则选 K、轮廓系数评估、MiniBatchKMeans

## 概述

KMeans 是聚类算法中的"Hello World"——简单、快速、应用广泛。算法只有两步迭代：**分配**（每个样本归到最近的质心）和**更新**（每个簇的质心更新为簇内均值）。收敛后，簇内方差最小化。

尽管假设簇为球形且大小相近（现实数据很难满足），KMeans 在很多实际场景中仍然是首选——因为它的计算效率高，且在"差不多"的数据上效果够好。

脚本演示了完整的 KMeans 工作流：训练、评估（轮廓系数、Calinski-Harabasz、ARI）、肘部法则选 K、KMeans++ 初始化和 MiniBatchKMeans。

## 关键代码解释

### 肘部法则（Elbow Method）

`python
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))
`

画出 K vs Inertia（簇内平方和）的曲线，找"肘部"——Inertia 下降速度突然变缓的点。配合轮廓系数（越高越好）一起选 K。

### KMeans++ 初始化

`python
kmeans_pp = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
`

KMeans++ 让初始质心尽可能分散：第一个随机选，后续的选择概率与到已有质心的距离成正比。这避免了糟糕的初始化导致收敛到局部最优。

### MiniBatchKMeans

`python
mbk = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42)
`

每次迭代只用一个 mini-batch 更新质心。速度比标准 KMeans 快很多，适合大数据集，质量略有下降。

## 使用示例

`python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
km = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = km.fit_predict(X)
print(f"Silhouette: {silhouette_score(X, labels):.4f}")
`

## 注意事项

1. **K 需要预设**：肘部法则、轮廓系数、Gap Statistic 都是选 K 的方法
2. **球形假设**：非凸形状的簇会被 KMeans 错误切分
3. **对初始化敏感**：
_init=10（默认）运行多次取最优
4. **必须特征缩放**：KMeans 基于欧氏距离
5. **n_init 参数已改为默认 10**：sklearn 新版本中默认值已更改

## 延伸思考

- **KMeans 的 EM 解释**：KMeans 可以看作 GMM 的特殊情况（等方差、硬分配）
- **Bisecting KMeans**：自顶向下的层次 KMeans
- **K-Medoids**：用实际样本点做中心（中位数），比 KMeans 对异常值更鲁棒
- **KMeans 的初始化变体**：K-Means||（Spark 中使用的并行初始化）