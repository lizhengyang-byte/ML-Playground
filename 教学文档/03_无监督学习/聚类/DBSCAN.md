# DBSCAN：发现任意形状簇的密度聚类

> 所属模块：03_无监督学习/聚类 | 源文件：DBSCAN.py | 核心功能：密度聚类、噪声识别、eps/min_samples 调参、k-距离图

## 概述

KMeans 假设簇是球形的，但现实中的数据形状千奇百怪——月牙形、环形、不规则形状。DBSCAN（Density-Based Spatial Clustering of Applications with Noise）基于密度而非距离来聚类：**密度高的区域形成簇，密度低的区域是噪声**。

核心概念：给定半径 eps 和最少点数 min_samples，如果一个点的 eps 邻域内有 >= min_samples 个点，它就是**核心点**。从核心点出发，通过密度可达性连接所有相邻的核心点和边界点，形成一个簇。

脚本在月牙形、同心圆和球形三种数据上测试了 DBSCAN，并演示了 k-距离图选 eps 的方法。

## 关键代码解释

### 参数搜索

`python
for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    for min_samples in [3, 5, 7, 10]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
`

DBSCAN 只有两个关键参数，但对结果影响很大。eps 太大 → 所有点合并成一个簇；太小 → 全是噪声点。

### k-距离图选 eps

`python
nn = NearestNeighbors(n_neighbors=5)
distances, _ = nn.kneighbors(X)
k_distances = np.sort(distances[:, -1])
`

对每个点计算到第 k 个最近邻的距离并排序。曲线的"拐点"就是合适的 eps 值——拐点之前是密集区域（距离小），之后是稀疏区域（距离突然增大）。

## 使用示例

`python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
`

## 注意事项

1. **噪声点 label=-1**：DBSCAN 会显式标记噪声点
2. **对 eps 极其敏感**：微小变化可能导致簇数剧变
3. **密度差异大的数据效果差**：不同密度的簇需要不同的 eps
4. **不预设簇数**：自动确定
5. **必须特征缩放**

## 延伸思考

- **HDBSCAN**：层次化 DBSCAN，自动处理不同密度的簇，只需 min_cluster_size 参数
- **OPTICS**：DBSCAN 的改进，不需要固定 eps，通过可达距离图自动确定聚类
- **DBSCAN 的噪声点用途**：可作为异常检测的副产品