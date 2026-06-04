# CatBoost：类别特征的最佳拍档
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


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
## 数学原理

### 1. CatBoost 的核心创新

CatBoost（Categorical Boosting）由 Yandex 提出，两大核心改进：
- **Ordered Target Statistics**：原生处理类别特征，避免目标泄漏
- **Ordered Boosting**：解决梯度提升中的预测偏移（prediction shift）

### 2. 类别特征的 Target Statistics

传统方法用目标均值编码类别特征：

$$\hat{x}_i^k = \frac{\sum_{j=1}^{n} [x_j^k = x_i^k] \cdot y_j}{\sum_{j=1}^{n} [x_j^k = x_i^k]}$$

这会导致**目标泄漏**：$y_i$ 参与了 $\hat{x}_i^k$ 的计算。

CatBoost 的解决方案（加先验平滑）：

$$\hat{x}_i^k = \frac{\sum_{j \in D_i} [x_j^k = x_i^k] \cdot y_j + a \cdot p}{\sum_{j \in D_i} [x_j^k = x_i^k] + a}$$

其中 $D_i = \{j : \sigma(j) < \sigma(i)\}$ 是随机排列中排在 $i$ 之前的样本，$p$ 是先验值，$a$ 是平滑系数。

### 3. Ordered Boosting

标准 Gradient Boosting 的预测偏移问题：

- 训练时用 $F_{m-1}$ 计算梯度 $g_i$
- 但 $F_{m-1}$ 是在包含 $x_i$ 的数据上训练的
- 导致梯度估计有偏

CatBoost 的解决：对每个样本 $x_i$，只用在**随机排列中排在 $i$ 之前**的样本来计算梯度：

$$g_i^{(m)} = -\frac{\partial L(y_i, F^{(m-1),\sigma(i)}(x_i))}{\partial F^{(m-1),\sigma(i)}(x_i)}$$

其中 $F^{(m-1),\sigma(i)}$ 只在 $\{j : \sigma(j) < \sigma(i)\}$ 上训练。

### 4. 对称决策树（Oblivious Trees）

CatBoost 使用**对称决策树**：所有同一层的节点使用相同的分裂条件。

- 每层的分裂特征和阈值相同
- 叶节点数为 $2^d$（$d$ 为深度）
- 分裂决策可编码为二进制索引，推理极快

$$\text{leaf}(x) = \sum_{l=0}^{d-1} 2^l \cdot \mathbb{I}[f_{k_l}(x) > t_l]$$

### 5. 推理效率

由于对称树的结构，推理时：
- 每个样本沿同一条路径分裂
- 可以用位运算高效计算叶节点编号
- 缓存命中率高

### 6. 类别特征的组合

CatBoost 还会自动探索类别特征的**组合**（combinations）：

$$c_{ij} = (c_i, c_j)$$

通过贪心策略寻找最优组合，自动发现特征交互。

### 7. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `CatBoostClassifier(iterations=100)` | $M=100$ 轮 Ordered Boosting |
| `depth=3` | 对称树深度 $d=3$，叶节点数 $2^3=8$ |
| `learning_rate=0.1` | 步长收缩 $\nu$ |
| `cat_features=["城市", "性别"]` | 类别特征，用 Ordered Target Statistics 编码 |
| `early_stopping_rounds=50` | 验证集 50 轮无提升则停止 |
| `best_iteration_` | 最优迭代次数 |
| `Pool` | CatBoost 的数据容器，存储类别特征元信息 |
