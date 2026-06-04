# DBSCAN：发现任意形状簇的密度聚类
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：03_无监督学习/聚类 | 源文件：DBSCAN.py | 核心功能：密度聚类、噪声识别、eps/min_samples 调参、k-距离图

## 概述

KMeans 假设簇是球形的，但现实中的数据形状千奇百怪——月牙形、环形、不规则形状。DBSCAN（Density-Based Spatial Clustering of Applications with Noise）基于密度而非距离来聚类：**密度高的区域形成簇，密度低的区域是噪声**。

核心概念：给定半径 eps 和最少点数 min_samples，如果一个点的 eps 邻域内有 >= min_samples 个点，它就是**核心点**。从核心点出发，通过密度可达性连接所有相邻的核心点和边界点，形成一个簇。

脚本在月牙形、同心圆和球形三种数据上测试了 DBSCAN，并演示了 k-距离图选 eps 的方法。

## 关键代码解释

### 参数搜索

```python
for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    for min_samples in [3, 5, 7, 10]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
```

DBSCAN 只有两个关键参数，但对结果影响很大。eps 太大 → 所有点合并成一个簇；太小 → 全是噪声点。

### k-距离图选 eps

```python
nn = NearestNeighbors(n_neighbors=5)
distances, _ = nn.kneighbors(X)
k_distances = np.sort(distances[:, -1])
```

对每个点计算到第 k 个最近邻的距离并排序。曲线的"拐点"就是合适的 eps 值——拐点之前是密集区域（距离小），之后是稀疏区域（距离突然增大）。

## 使用示例

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
```

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
## 数学原理

### 1. 核心概念的数学定义

**代码对应**：`DBSCAN(eps=eps, min_samples=min_samples)` 中的两个关键参数。

设 $\varepsilon$ 为邻域半径，$\text{MinPts}$ 为最小样本数：

**$\varepsilon$-邻域**：$N_\varepsilon(\mathbf{x}) = \{\mathbf{x}_i : \|\mathbf{x}_i - \mathbf{x}\| \leq \varepsilon\}$

**核心点**（Core Point）：$|N_\varepsilon(\mathbf{x})| \geq \text{MinPts}$

**边界点**（Border Point）：不是核心点，但在某个核心点的 $\varepsilon$-邻域内

**噪声点**（Noise）：既不是核心点也不是边界点

### 2. DBSCAN 算法流程

1. 对每个未访问的点 $\mathbf{x}$：
   - 计算 $N_\varepsilon(\mathbf{x})$
   - 如果 $|N_\varepsilon(\mathbf{x})| < \text{MinPts}$，暂时标记为噪声（后续可能变为边界点）
   - 如果 $|N_\varepsilon(\mathbf{x})| \geq \text{MinPts}$（核心点）：
     - 创建新簇 $C$，将 $\mathbf{x}$ 加入
     - **密度可达扩展**：将 $N_\varepsilon(\mathbf{x})$ 中所有点加入队列
     - 对队列中每个核心点，将其 $\varepsilon$-邻域中的点也加入簇 $C$
     - 重复直到没有新点可加入

### 3. 密度连接的数学关系

**直接密度可达**（Directly Density-Reachable）：$\mathbf{x}$ 是核心点，$\mathbf{y} \in N_\varepsilon(\mathbf{x})$

**密度可达**（Density-Reachable）：存在链 $\mathbf{x}_1, \ldots, \mathbf{x}_n$，其中 $\mathbf{x}_{i+1}$ 从 $\mathbf{x}_i$ 直接密度可达

**密度相连**（Density-Connected）：存在 $\mathbf{o}$ 使得 $\mathbf{x}$ 和 $\mathbf{y}$ 都从 $\mathbf{o}$ 密度可达

一个簇 $C$ 满足：
- $\forall \mathbf{x} \in C, \mathbf{y}$ 密度可达自 $\mathbf{x} \Rightarrow \mathbf{y} \in C$（最大化性）
- $\forall \mathbf{x}, \mathbf{y} \in C$，$\mathbf{x}$ 和 $\mathbf{y}$ 密度相连（连通性）

### 4. k-距离图选 eps

**代码对应**：代码中使用 `NearestNeighbors(n_neighbors=5)` 计算 k-距离。

对每个点计算到第 $k$ 个最近邻的距离，排序后绘制 k-距离图。**拐点**（elbow）处的距离即为合适的 $\varepsilon$：

- $\varepsilon$ 太小：大部分点成为噪声
- $\varepsilon$ 太大：所有点合并为一个簇
- 拐点处：密度变化最显著的边界

### 5. 时间复杂度与空间复杂度

- **朴素实现**：$O(n^2)$（每个点需计算与所有其他点的距离）
- **使用空间索引**（KD-Tree、Ball Tree）：$O(n\log n)$（低维时有效）
- **空间复杂度**：$O(n)$ 存储标签和邻域信息

### 6. DBSCAN 的局限性

- **密度不均匀**：如果各簇密度差异大，单一的 $\varepsilon$ 无法同时捕捉所有簇
- **高维数据**：高维空间中距离趋于相同（维度灾难），$\varepsilon$ 的选择变得困难
- **参数敏感**：$\varepsilon$ 和 MinPts 的微小变化可能导致结果截然不同

改进算法：HDBSCAN（层次 DBSCAN）自动处理不同密度的簇。
