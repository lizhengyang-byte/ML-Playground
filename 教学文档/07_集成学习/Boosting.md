# Boosting：串行训练、逐步纠错

> 所属模块：07_集成学习 | 源文件：Boosting.py | 核心功能：串行集成、残差拟合、偏差降低

## 概述

Boosting 与 Bagging 的核心区别：基模型**串行**训练，每个新模型专注于修正前一个模型的错误。核心价值：**降低偏差**。

## 关键代码解释

```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
```

`learning_rate` 控制每棵树的贡献权重（缩小步长，防止过拟合）。

## 注意事项

1. 不能并行（串行依赖）
2. 容易过拟合（需控制 n_estimators 和 learning_rate）
3. learning_rate 和 n_estimators 通常反向调节

## 延伸思考

- **XGBoost/LightGBM/CatBoost**：工业级 Boosting 实现
- **AdaBoost**：最早的 Boosting 算法
- **学习率调度**：逐步降低学习率