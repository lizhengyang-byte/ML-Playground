# CatBoost：类别特征的最佳拍档

> 所属模块：07_集成学习 | 源文件：CatBoost.py | 核心功能：Ordered Target Encoding、对称树、原生类别特征

## 概述

CatBoost 是 Yandex 开发的梯度提升框架，最大特点是**原生处理类别特征**——不需要手动编码。它用 Ordered Target Encoding 替代传统 Target Encoding，避免了数据泄露。

## 关键代码解释

```python
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                           cat_features=[0, 1, 2], verbose=50)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)
```

## 注意事项

1. **类别特征索引**：需要指定哪些列是类别特征
2. **对称树**：CatBoost 使用对称决策树，推理更快
3. **默认参数通常就够好**：调参需求比 XGBoost/LightGBM 少

## 延伸思考

- **Ordered Boosting**：减少梯度估计的偏差
- **CatBoost for ranking**：排序任务的内置支持
- **三者对比**：XGBoost（稳定）/ LightGBM（快）/ CatBoost（类别特征好）