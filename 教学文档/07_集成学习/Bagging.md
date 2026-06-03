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
﻿## 数学原理

### 1. Bootstrap 采样

Bagging 的核心是 **Bootstrap 自助采样**：从大小为 $n$ 的训练集 $D$ 中，有放回地随机抽取 $n$ 个样本组成子集 $D_b$。

每个样本在一次 Bootstrap 采样中**未被选中**的概率为：

$$\left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368$$

即每个基学习器平均只使用约 $63.2\%$ 的训练样本，剩余 $36.8\%$ 称为 **袋外样本（Out-of-Bag, OOB）**。

### 2. 方差缩减原理

设基学习器的预测方差为 $\text{Var}(f)$，预测之间的相关系数为 $\rho$，则 $B$ 个基学习器平均后的方差为：

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B}f_b\right) = \rho \cdot \text{Var}(f) + \frac{1-\rho}{B}\cdot \text{Var}(f)$$

- $\rho = 0$（完全独立）：方差缩小 $B$ 倍
- $\rho = 1$（完全相同）：无方差缩减
- 通过**随机性**（Bootstrap 采样 + 特征子采样）降低 $\rho$

### 3. 分类：多数投票

分类任务中，Bagging 的最终预测为：

$$\hat{y} = \arg\max_c \sum_{b=1}^{B} \mathbb{I}[f_b(x) = c]$$

其中 $\mathbb{I}[\cdot]$ 是指示函数。代码中 `BaggingClassifier` 默认使用硬投票。

### 4. 回归：均值聚合

回归任务中，最终预测为所有基学习器的简单平均：

$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B} f_b(x)$$

### 5. OOB 误差估计

OOB 样本提供了一种**无需额外验证集**的泛化误差估计：

$$\text{OOB Error} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{I}\left[y_i \neq \text{mode}\{f_b(x_i) : i \notin D_b\}\right]$$

对每个样本 $x_i$，只用那些**未包含** $x_i$ 的基学习器进行预测，取多数投票。

代码中 `oob_score=True` 计算此指标。

### 6. Bagging vs Pasting

| 采样方式 | 有放回 | 特点 |
|----------|--------|------|
| Bagging | 是 | 基学习器有差异，可用 OOB 评估 |
| Pasting | 否 | 基学习器更独立，但无 OOB |

代码中 `bootstrap=True` 为 Bagging，`bootstrap=False` 为 Pasting。

### 7. 随机子空间（特征采样）

代码中 `max_features=0.8` 实现**随机子空间法**：每个基学习器只使用 $80\%$ 的特征。结合样本采样，形成 **Random Patches** 方法，进一步降低 $\rho$。

### 8. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `n_estimators=50` | 集成 $B=50$ 个基学习器 |
| `max_samples=0.8` | 每次 Bootstrap 采样 $0.8n$ 个样本 |
| `max_features=0.8` | 每次随机选取 $80\%$ 特征（随机子空间） |
| `bootstrap=True` | 有放回采样（Bagging） |
| `oob_score=True` | 计算袋外误差估计泛化性能 |
| `BaggingRegressor` | 回归聚合：$\hat{y} = \frac{1}{B}\sum f_b(x)$ |
| `BaggingClassifier` | 分类聚合：$\hat{y} = \arg\max_c \sum \mathbb{I}[f_b(x)=c]$ |
