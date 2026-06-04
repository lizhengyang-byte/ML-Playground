# AdaBoost：加权投票的 Boosting 先驱
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：07_集成学习 | 源文件：AdaBoost.py | 核心功能：样本权重调整、弱分类器加权组合

## 概述

AdaBoost（Adaptive Boosting）是最早的 Boosting 算法。每轮训练后，增加被错误分类样本的权重，让下一个模型更关注"难"样本。

## 关键代码解释

```python
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm="SAMME")
```

每个弱分类器有一个权重 alpha = 0.5 * ln((1-err)/err)，准确率越高权重越大。

## 注意事项

1. 对噪声和异常值敏感（权重会被难样本主导）
2. 基分类器不能太强（通常用决策树桩 depth=1）
3. SAMME vs SAMME.R：SAMME.R 用概率输出，通常更好

## 延伸思考

- **AdaBoost.R2**：回归版本
- **Gradient Boosting**：更通用的 Boosting 框架
- **AdaBoost 与 SVM 的联系**：两者都关注"难"样本
## 数学原理

### 1. AdaBoost 算法概述

AdaBoost（Adaptive Boosting）通过**调整样本权重**来关注难分类样本。每一轮训练后，增加被错误分类样本的权重，减少正确分类样本的权重。

### 2. 样本权重初始化

设训练集 $\{(x_i, y_i)\}_{i=1}^n$，$y_i \in \{-1, +1\}$。初始权重：

$$w_i^{(1)} = \frac{1}{n}, \quad i=1,\ldots,n$$

### 3. 第 $m$ 轮的迭代过程

**步骤一**：用加权训练集训练弱学习器 $f_m(x)$

**步骤二**：计算加权错误率：

$$\epsilon_m = \frac{\sum_{i=1}^{n} w_i^{(m)} \mathbb{I}[y_i \neq f_m(x_i)]}{\sum_{i=1}^{n} w_i^{(m)}}$$

**步骤三**：计算学习器权重（$\alpha_m$）：

$$\alpha_m = \frac{1}{2}\ln\left(\frac{1-\epsilon_m}{\epsilon_m}\right)$$

- 当 $\epsilon_m < 0.5$（比随机猜测好）：$\alpha_m > 0$
- 当 $\epsilon_m = 0$（完美分类）：$\alpha_m \to \infty$
- 当 $\epsilon_m \geq 0.5$（不比猜测好）：终止

**步骤四**：更新样本权重：

$$w_i^{(m+1)} = w_i^{(m)} \cdot \exp\left(-\alpha_m y_i f_m(x_i)\right)$$

归一化使 $\sum w_i = 1$。被错误分类的样本（$y_i f_m(x_i) < 0$）权重增大。

### 4. 最终分类器

$$F(x) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m f_m(x)\right)$$

这是 $M$ 个弱学习器的**加权投票**，权重 $\alpha_m$ 由每个学习器的准确度决定。

### 5. 指数损失函数

AdaBoost 等价于在加法模型上最小化**指数损失**：

$$L(y, F(x)) = \exp(-y F(x))$$

对 $F$ 求导：

$$\frac{\partial L}{\partial F} = -y \exp(-y F(x))$$

每一步通过前向分步加法模型拟合负梯度，推导出 AdaBoost 的权重更新公式。

### 6. 学习率的作用

代码中 `learning_rate` 用于正则化 $\alpha_m$：

$$F(x) = \sum_{m=1}^{M} \nu \cdot \alpha_m f_m(x), \quad 0 < \nu \leq 1$$

较小的 $\nu$ 使每个弱学习器的贡献缩小，需要更多轮次但泛化更好。

### 7. 基学习器复杂度

代码对比了不同树深度（1, 2, 3, 5）的效果：
- 深度=1（决策树桩）：最简单的弱学习器，AdaBoost 的经典选择
- 深度增大 → 单个学习器更强 → $\epsilon_m$ 更小 → $\alpha_m$ 更大
- 但过强的基学习器可能导致过拟合

### 8. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `AdaBoostClassifier(n_estimators=50)` | $M=50$ 轮迭代 |
| `learning_rate=1.0` | $\nu=1$，不对 $\alpha_m$ 缩放 |
| `DecisionTreeClassifier(max_depth=1)` | 决策树桩作为弱学习器 $f_m$ |
| 样本权重（隐式） | $w_i^{(m)}$，由 `sklearn` 内部管理 |
| `feature_importances_` | $\sum_m \alpha_m \cdot (\text{特征在 } f_m \text{ 中的重要性})$ |
