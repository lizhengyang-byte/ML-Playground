# 均值漂移 Mean Shift：跟着密度峰值走
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：03_无监督学习/聚类 | 源文件：均值漂移_MeanShift.py | 核心功能：核密度估计聚类、bandwidth 自动估计、无需预设簇数

## 概述

均值漂移（Mean Shift）的核心思想非常直观：每个点都向它邻域内所有点的均值方向"漂移"，直到收敛到**密度峰值**。最终收敛到同一峰值的点被归为一簇。

与 KMeans 的区别：KMeans 需要预设 K 且收敛到指定数量的质心，Mean Shift 自动确定簇数——有多少个密度峰值就有多少个簇。

脚本在月牙形和球形数据上演示了 Mean Shift 的使用，bandwidth 自动估计和不同带宽的影响。

## 关键代码解释

### bandwidth 自动估计

```python
bw_estimated = estimate_bandwidth(X, quantile=0.2, n_samples=200)
```

bandwidth 是 Mean Shift 唯一的关键参数，控制核函数的"视野"大小。estimate_bandwidth 用分位数方法自动估计。quantile 越大，带宽越大，簇数越少。

### bin_seeding 加速

```python
ms = MeanShift(bandwidth=bw_estimated, bin_seeding=True)
```

in_seeding=True 只在网格点上初始化漂移起点，大幅减少计算量（速度提升数倍），结果几乎不变。

## 使用示例

```python
from sklearn.cluster import MeanShift, estimate_bandwidth
bw = estimate_bandwidth(X, quantile=0.2)
ms = MeanShift(bandwidth=bw, bin_seeding=True)
labels = ms.fit_predict(X)
```

## 注意事项

1. **bandwidth 是唯一参数但极其关键**：太大 → 一个簇，太小 → 每个点一个簇
2. **时间复杂度 O(n²)**：大数据集很慢
3. **密度均匀时效果好**：密度差异大时可能无法正确分离
4. **不适合高维数据**：核密度估计在高维空间退化

## 延伸思考

- **Mean Shift 追踪**：计算机视觉中的目标跟踪算法（CamShift）
- **核密度估计（KDE）**：Mean Shift 的理论基础
- **Mean Shift vs DBSCAN**：两者都能发现任意形状的簇，但原理不同
- **并行 Mean Shift**：GPU 加速版本
## 数学原理

### 1. 核密度估计（KDE）

**代码对应**：`MeanShift(bandwidth=bw)` 中的 bandwidth 参数。

均值漂移基于**核密度估计**。给定数据 $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$，密度估计为：

$$\hat{f}(\mathbf{x}) = \frac{1}{n h^p}\sum_{i=1}^{n}K\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)$$

其中 $h$ 为带宽（bandwidth），$K$ 为核函数（通常为高斯核）。

### 2. 均值漂移向量

对密度估计求梯度：

$$\nabla\hat{f}(\mathbf{x}) = \frac{1}{n h^{p+2}}\sum_{i=1}^{n}(\mathbf{x}_i - \mathbf{x})K'\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)$$

均值漂移向量为梯度的归一化方向：

$$\mathbf{m}(\mathbf{x}) = \frac{\sum_{i=1}^{n}\mathbf{x}_i g\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)}{\sum_{i=1}^{n}g\left(\frac{\mathbf{x} - \mathbf{x}_i}{h}\right)} - \mathbf{x}$$

其中 $g(x) = -K'(x)$。$\mathbf{m}(\mathbf{x})$ 指向密度增长最快的方向。

### 3. 算法流程

1. 对每个样本 $\mathbf{x}_i$，计算其均值漂移向量 $\mathbf{m}(\mathbf{x}_i)$
2. 将 $\mathbf{x}_i$ 移动到 $\mathbf{x}_i + \mathbf{m}(\mathbf{x}_i)$
3. 重复直到收敛到**模式点**（密度局部最大值）
4. 收敛到同一模式点的样本归为同一簇

### 4. 带宽选择

带宽 $h$ 是最关键的参数：
- $h$ 大：密度估计平滑，簇数少（欠拟合）
- $h$ 小：密度估计粗糙，簇数多（过拟合）

sklearn 提供 `estimate_bandwidth(X, quantile=q)` 自动估计带宽，基于 $q$ 分位数的最近邻距离。

### 5. 与 KMeans 的对比

| 特性 | KMeans | MeanShift |
|------|--------|-----------|
| 簇数 | 需预设 $K$ | 自动确定 |
| 簇形状 | 球形 | 任意形状（取决于带宽） |
| 优化目标 | 最小化 WCSS | 寻找密度模式 |
| 复杂度 | $O(nKdT)$ | $O(n^2 T)$ |
| 参数 | $K$ | bandwidth $h$ |
