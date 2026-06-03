# Bagging：并行训练、投票决策

> 所属模块：07_集成学习 | 源文件：Bagging.py | 核心功能：Bootstrap 采样、并行集成、方差降低

## 概述

Bagging（Bootstrap Aggregating）是最基础的集成方法：通过 Bootstrap（有放回抽样）生成多个训练子集，在每个子集上训练一个基模型，最终取平均（回归）或投票（分类）。核心价值：**降低方差**，不改变偏差。

## 关键代码解释

```python
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(n_estimators=50, max_samples=0.8, max_features=0.8)
bag.fit(X_train, y_train)
```

`max_samples` 和 `max_features` 控制每个基模型看到的数据和特征比例。

## 注意事项

1. 对高偏差模型（如浅层决策树）效果有限
2. 对高方差模型（如深层决策树）效果显著
3. 各基模型独立训练，可完全并行

## 延伸思考

- **随机森林** = Bagging + 随机特征选择
- **Pasting**：不放回抽样的 Bagging
- **Subagging**：只用部分样本的 Bagging