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