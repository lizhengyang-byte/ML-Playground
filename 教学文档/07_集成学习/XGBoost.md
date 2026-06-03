# XGBoost：竞赛神器

> 所属模块：07_集成学习 | 源文件：XGBoost.py | 核心功能：梯度提升、正则化、特征重要性、早停

## 概述

XGBoost（eXtreme Gradient Boosting）是梯度提升的高效实现，加了正则化项防止过拟合。在 Kaggle 竞赛中长期霸榜，是结构化数据的首选模型之一。

## 关键代码解释

```python
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
```

## 注意事项

1. **早停**：监控验证集性能，连续 N 轮不提升就停止
2. **学习率与树数反向调节**：lr 小则 n_estimators 需要大
3. **GPU 加速**：`tree_method="hist"` + `device="cuda"`

## 延伸思考

- **XGBoost 1.0+**：支持原生分类特征
- **DART**：在 XGBoost 中加入 Dropout
- **XGBoost vs LightGBM**：后者通常更快，但 XGBoost 更稳定