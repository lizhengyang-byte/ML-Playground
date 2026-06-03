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
﻿## 数学原理

### 1. LightGBM 的核心创新

LightGBM（Light Gradient Boosting Machine）由微软提出，核心改进是**基于直方图的决策树算法**和**叶子优先生长策略**，大幅提升训练速度。

### 2. 直方图算法（Histogram-based Splitting）

传统 XGBoost 需要遍历所有特征值寻找最优分裂点，LightGBM 将连续特征离散化为 $k$ 个桶（bin）：

1. 对每个特征，将 $n$ 个样本值映射到 $k$ 个等频直方图桶中
2. 遍历 $k$ 个桶的分裂点（而非 $n$ 个值），时间复杂度从 $O(n \cdot d)$ 降至 $O(k \cdot d)$

$$\text{split cost} = O(k \cdot d) \ll O(n \cdot d), \quad k \ll n$$

### 3. 叶子优先生长（Leaf-wise Growth）

LightGBM 采用**叶子优先**策略（而非层序生长）：

$$j^* = \arg\max_{j \in \text{leaves}} \text{Gain}(j)$$

每步选择**增益最大的叶子节点**进行分裂，直到达到 `num_leaves` 限制。

与 XGBoost 的层序生长相比：
- 同样的 `num_leaves` 下，叶子优先能构建更深、更不对称的树
- 损失下降更快，但需要注意过拟合

### 4. Gradient-based One-Side Sampling (GOSS)

LightGBM 对样本进行智能采样：
- 保留梯度绝对值大的样本（信息量大）
- 随机丢弃梯度小的样本（已学好的样本）

设 $a$ 为大梯度样本比例，$b$ 为小梯度样本比例：

$$\tilde{g}_j = \frac{1}{n}\left(\sum_{x_i \in A_l} g_i + \frac{1-a}{b}\sum_{x_i \in B_l} g_i\right)$$

其中 $A_l$ 是大梯度集合，$B_l$ 是小梯度的随机子集，$\frac{1-a}{b}$ 是补偿系数。

### 5. Exclusive Feature Bundling (EFB)

高维稀疏数据中，许多特征是互斥的（不同时取非零值）。LightGBM 将互斥特征捆绑为一个特征，降低有效特征数：

$$\text{bundles} \ll \text{features}$$

这在高维稀疏数据中效果显著。

### 6. num_leaves 与 max_depth 的关系

LightGBM 的核心复杂度控制参数是 `num_leaves`（叶子数），而非 `max_depth`：

$$\text{num\_leaves} \leq 2^{\text{max\_depth}}$$

- `num_leaves=31`：最多 31 个叶节点，模型复杂度适中
- `max_depth` 作为辅助约束，`-1` 表示不限制深度

### 7. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `num_leaves=31` | 叶子优先生长直到 31 个叶子 |
| `max_depth=3` | 深度约束（辅助限制） |
| `learning_rate=0.1` | 步长收缩 $\nu$ |
| `subsample=0.8` | 行采样比例 |
| `colsample_bytree=0.8` | 列采样比例（随机子空间） |
| `min_data_in_leaf` | 叶节点最小样本数，防过拟合 |
| `early_stopping(50)` | 验证集 50 轮无提升则停止 |
| `best_iteration_` | 最优迭代次数 |
