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
﻿## 数学原理

### 1. XGBoost 的目标函数

XGBoost 在第 $t$ 轮最小化以下目标函数（包含正则项）：

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

其中正则项：

$$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2$$

$T$ 是叶子节点数，$w_j$ 是叶节点权重。

### 2. 泰勒展开近似

对损失函数在 $\hat{y}_i^{(t-1)}$ 处二阶泰勒展开：

$$L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t(x_i)^2$$

其中一阶和二阶梯度统计量：

$$g_i = \frac{\partial L(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}, \quad h_i = \frac{\partial^2 L(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$$

### 3. 最优叶节点权重

对每个叶节点 $j$，最优权重为：

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

其中 $I_j$ 是落入叶节点 $j$ 的样本集合。

### 4. 最优增益

分裂一个叶节点的增益：

$$\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda}\right] - \gamma$$

- $\gamma$ 控制分裂的门槛，自动实现剪枝
- Gain $< 0$ 时不进行分裂

### 5. XGBoost 的关键优化

| 优化技术 | 说明 |
|----------|------|
| 列采样（`colsample_bytree`） | 每棵树只用部分特征，类似随机森林 |
| 行采样（`subsample`） | 每棵树只用部分样本 |
| 加权分位数（Weighted Quantile Sketch） | 高效寻找近似分裂点 |
| 缓存感知访问 | 减少缓存未命中 |
| 稀疏感知 | 自动处理缺失值 |

### 6. 与标准 Gradient Boosting 的区别

| 方面 | Gradient Boosting | XGBoost |
|------|-------------------|---------|
| 梯度 | 仅用一阶梯度 $g_i$ | 用一阶 $g_i$ + 二阶 $h_i$ |
| 正则化 | 无显式正则 | $\gamma T + \frac{1}{2}\lambda\|w\|^2$ |
| 树结构 | 贪心分裂 | 带增益剪枝 |
| 并行化 | 串行 | 特征级别并行 |

### 7. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `max_depth=3` | 树的最大深度，控制模型复杂度 |
| `learning_rate=0.1` | 步长收缩 $\nu$，$F_t = F_{t-1} + \nu f_t$ |
| `subsample=0.8` | 行采样，每轮用 $80\%$ 样本 |
| `colsample_bytree=0.8` | 列采样，每棵树用 $80\%$ 特征 |
| `eval_metric="logloss"` | 对数损失 $L = -[y\log p + (1-y)\log(1-p)]$ |
| `feature_importances_` | 基于增益（gain）的特征重要性 |
| `GridSearchCV` | 超参数搜索：$\max\_depth, \nu, M$ |
