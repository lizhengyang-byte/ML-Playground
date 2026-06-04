# KMeans 聚类：最简单也最常用的聚类算法
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：03_无监督学习/聚类 | 源文件：KMeans.py | 核心功能：KMeans 聚类、肘部法则选 K、轮廓系数评估、MiniBatchKMeans

## 概述

KMeans 是聚类算法中的"Hello World"——简单、快速、应用广泛。算法只有两步迭代：**分配**（每个样本归到最近的质心）和**更新**（每个簇的质心更新为簇内均值）。收敛后，簇内方差最小化。

尽管假设簇为球形且大小相近（现实数据很难满足），KMeans 在很多实际场景中仍然是首选——因为它的计算效率高，且在"差不多"的数据上效果够好。

脚本演示了完整的 KMeans 工作流：训练、评估（轮廓系数、Calinski-Harabasz、ARI）、肘部法则选 K、KMeans++ 初始化和 MiniBatchKMeans。

## 关键代码解释

### 肘部法则（Elbow Method）

```python
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))
```

画出 K vs Inertia（簇内平方和）的曲线，找"肘部"——Inertia 下降速度突然变缓的点。配合轮廓系数（越高越好）一起选 K。

### KMeans++ 初始化

```python
kmeans_pp = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
```

KMeans++ 让初始质心尽可能分散：第一个随机选，后续的选择概率与到已有质心的距离成正比。这避免了糟糕的初始化导致收敛到局部最优。

### MiniBatchKMeans

```python
mbk = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42)
```

每次迭代只用一个 mini-batch 更新质心。速度比标准 KMeans 快很多，适合大数据集，质量略有下降。

## 使用示例

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
km = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = km.fit_predict(X)
print(f"Silhouette: {silhouette_score(X, labels):.4f}")
```

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
## 数学原理

### 1. KMeans 的目标函数

**代码对应**：`KMeans(n_clusters=4, n_init=10)` 训练 KMeans。

KMeans 最小化**簇内平方和**（Within-Cluster Sum of Squares, WCSS）：

$$J = \sum_{k=1}^{K}\sum_{\mathbf{x}_i \in C_k}\|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2^2$$

其中 $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k}\mathbf{x}_i$ 为簇 $k$ 的质心。

**代码对应**：`kmeans.inertia_` 返回 $J$ 的值（惯性）。

### 2. Lloyd 算法（交替优化）

KMeans 通过交替优化求解（非凸问题，只能保证局部最优）：

**步骤 1 — 分配**：将每个样本分配到最近的质心：

$$C_k = \{\mathbf{x}_i : \|\mathbf{x}_i - \boldsymbol{\mu}_k\| \leq \|\mathbf{x}_i - \boldsymbol{\mu}_j\|, \forall j \neq k\}$$

**步骤 2 — 更新**：重新计算每个簇的质心：

$$\boldsymbol{\mu}_k \leftarrow \frac{1}{|C_k|}\sum_{\mathbf{x}_i \in C_k}\mathbf{x}_i$$

两步交替直到收敛（质心不再变化或达到最大迭代次数）。

**代码对应**：`kmeans.n_iter_` 返回实际迭代次数。

### 3. KMeans++ 初始化

**代码对应**：`init="k-means++"` 使用更智能的初始化策略。

随机初始化可能导致很差的质心选择。KMeans++ 使初始质心尽可能分散：

1. 随机选择第一个质心 $\boldsymbol{\mu}_1$
2. 对每个样本 $\mathbf{x}_i$，计算到最近已选质心的距离 $D(\mathbf{x}_i)^2$
3. 以概率 $P(\boldsymbol{\mu}_{\text{next}} = \mathbf{x}_i) = D(\mathbf{x}_i)^2 / \sum_j D(\mathbf{x}_j)^2$ 选择下一个质心
4. 重复直到选满 $K$ 个质心

KMeans++ 保证初始化质量：期望的初始 WCSS 最多为最优解的 $O(\ln K)$ 倍。

### 4. 轮廓系数

**代码对应**：`silhouette_score(X, labels)` 计算轮廓系数。

对样本 $i$，设 $a(i)$ 为 $i$ 到同簇其他样本的平均距离（簇内不相似度），$b(i)$ 为 $i$ 到最近其他簇所有样本的平均距离（最近簇不相似度）：

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

$s(i) \in [-1, 1]$：
- $s(i) \approx 1$：样本被正确聚类
- $s(i) \approx 0$：样本在簇边界
- $s(i) < 0$：样本可能被分错了簇

### 5. 肘部法则

**代码对应**：代码中 `for k in range(2, 11)` 计算了不同 K 的惯性。

惯性 $J(K)$ 随 $K$ 增大而单调递减（$K = N$ 时 $J = 0$）。肘部法则寻找 $J(K)$ 曲线的"拐点"——$J$ 下降速度突然变缓的位置。

数学上，可以计算 $J(K)$ 的二阶差分：

$$\Delta^2 J(K) = J(K+1) - 2J(K) + J(K-1)$$

$\Delta^2 J$ 最大的 $K$ 即为拐点。

### 6. 时间复杂度

KMeans 每次迭代需要计算所有 $n$ 个样本到 $K$ 个质心的距离：

$$O(n \cdot K \cdot d \cdot T)$$

其中 $d$ 为特征数，$T$ 为迭代次数。MiniBatchKMeans 通过每次只用 $B$ 个样本更新质心，将复杂度降为 $O(B \cdot K \cdot d \cdot T)$。
