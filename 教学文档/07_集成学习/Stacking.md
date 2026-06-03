# Stacking：用元模型学习如何组合

> 所属模块：07_集成学习 | 源文件：Stacking.py | 核心功能：两层集成、元学习器、交叉验证预测

## 概述

Stacking 用第一层多个基模型的预测作为特征，训练一个元模型（meta-learner）做最终预测。比简单投票更灵活——元模型可以学习"哪些基模型在什么情况下更可信"。

## 关键代码解释

```python
from sklearn.ensemble import StackingClassifier
stacking = StackingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("svm", svm)],
    final_estimator=LogisticRegression(), cv=5
)
```

`cv=5` 确保元特征不泄露：每个样本的元特征来自它未参与训练的 fold 的预测。

## 注意事项

1. 必须用交叉验证生成元特征，否则严重过拟合
2. 元模型通常用简单模型（逻辑回归）
3. 训练时间 = 基模型数 × CV 折数 + 元模型训练

## 延伸思考

- **多层 Stacking**：Stacking 的输出再做 Stacking
- **Feature-weighted Stacking**：给不同特征组用不同基模型
- **Netflix Prize**：冠军方案就是大规模 Stacking
﻿## 数学原理

### 1. Stacking 的两层结构

Stacking（堆叠泛化）由两层组成：

- **第一层**：$K$ 个不同的基学习器 $f_1, f_2, \ldots, f_K$
- **第二层**：一个**元学习器** $g$，学习如何最优地组合基学习器的预测

$$\hat{y} = g(f_1(x), f_2(x), \ldots, f_K(x))$$

### 2. 元特征的生成（K 折交叉预测）

为避免信息泄漏，第一层的训练使用 **K 折交叉预测**（cross_val_predict）：

对每个基学习器 $f_k$ 和每个折 $j=1,\ldots,J$：
1. 在 $J-1$ 个折上训练 $f_k$
2. 对第 $j$ 折的样本预测，得到 $\hat{y}_k^{(j)}$
3. 拼接所有折的预测，形成**元特征** $z_k = (\hat{y}_k^{(1)}, \ldots, \hat{y}_k^{(n)})$

代码中：
```python
meta_features_train.append(
    cross_val_predict(model, X_train, y_train, cv=5, method="predict")
)
```

### 3. 元特征矩阵

将所有基学习器的预测堆叠为元特征矩阵：

$$Z = [z_1, z_2, \ldots, z_K] \in \mathbb{R}^{n \times K}$$

- 分类任务：每列是类别标签（或概率向量，维度为 $K \times C$）
- 回归任务：每列是连续预测值

代码中 `np.column_stack(meta_features_train)` 完成此操作。

### 4. 元学习器的训练

在元特征矩阵 $Z$ 上训练元学习器 $g$：

$$\hat{g} = \arg\min_g \sum_{i=1}^{n} L(y_i, g(z_i))$$

代码中使用 `LogisticRegression` 作为元学习器。元学习器学到的是：在什么情况下信任哪个基学习器。

### 5. 测试集预测流程

1. 每个基学习器在**全部训练集**上重新训练
2. 对测试集预测，得到元特征 $Z_{test}$
3. 元学习器在 $Z_{test}$ 上预测最终结果

```python
for name, model in base_estimators:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    meta_features_test.append(pred)
meta_X_test = np.column_stack(meta_features_test)
y_pred = meta_model.predict(meta_X_test)
```

### 6. Stacking 的数学优势

设基学习器的误差为 $\epsilon_k$，若它们的错误**不相关**，元学习器可以通过加权组合降低总误差：

$$\text{Var}\left(\sum_k w_k f_k\right) = \sum_k w_k^2 \text{Var}(f_k) + 2\sum_{k<l} w_k w_l \text{Cov}(f_k, f_l)$$

多样性高的基学习器（低协方差）组合效果更好。

### 7. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `cross_val_predict(..., cv=5)` | K=5 折交叉预测生成元特征 $z_k$ |
| `np.column_stack(meta_features)` | 构建元特征矩阵 $Z \in \mathbb{R}^{n \times K}$ |
| `LogisticRegression()` 作为 meta_model | 元学习器 $g(z)$ |
| `method="predict"` | 元特征为类别标签（离散值） |
| 5 个不同的基学习器 | $f_1, \ldots, f_5$，多样性是 Stacking 的关键 |
| 基学习器全部重新 fit | 测试时用完整训练数据获得最佳预测 |
