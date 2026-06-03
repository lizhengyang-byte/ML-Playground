# LightGBM：大规模梯度提升的利器

> 所属模块：07_集成学习 | 源文件：LightGBM.py | 核心功能：直方图算法、Leaf-wise 生长、类别特征支持

## 概述

LightGBM 是微软开源的梯度提升框架。两大创新：GOSS（基于梯度的单边采样）和 EFB（互斥特征捆绑），训练速度比 XGBoost 快数倍。

## 关键代码解释

```python
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10)])
```

`num_leaves` 控制树的复杂度（LightGBM 用 Leaf-wise 生长，不同于 XGBoost 的 Level-wise）。

## 注意事项

1. **num_leaves** 是最重要的参数，通常 < 2^max_depth
2. **小数据集容易过拟合**：需要限制 min_data_in_leaf
3. **原生支持分类特征**：不需要 one-hot 编码

## 延伸思考

- **GOSS**：优先保留梯度大的样本
- **EFB**：将互斥特征捆绑，减少特征数
- **Dask-LightGBM**：分布式训练