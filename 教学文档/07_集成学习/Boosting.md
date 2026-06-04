# Boosting：串行训练、逐步纠错
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


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
## 数学原理

### 1. Boosting 的基本思想

Boosting 将多个**弱学习器**（性能略优于随机猜测）串行组合为**强学习器**。核心策略：每一轮重点关注前一轮预测错误的样本。

### 2. 加法模型（Additive Model）

Boosting 构建的集成模型是加法展开：

$$F_M(x) = \sum_{m=1}^{M} \alpha_m f_m(x)$$

其中 $f_m$ 是第 $m$ 轮的基学习器，$\alpha_m$ 是其权重，$M$ 是总轮数。

### 3. 梯度提升（Gradient Boosting）

Gradient Boosting 的核心思想：每一轮拟合**损失函数的负梯度**（伪残差）。

设损失函数为 $L(y, F(x))$，第 $m$ 轮的伪残差：

$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}, \quad i=1,\ldots,n$$

**二分类**（对数损失）：$L = -[y\log p + (1-y)\log(1-p)]$，伪残差为 $r_{im} = y_i - p_{m-1}(x_i)$。

**回归**（MSE）：$L = \frac{1}{2}(y-F)^2$，伪残差为 $r_{im} = y_i - F_{m-1}(x_i)$（即当前残差）。

### 4. 基学习器拟合伪残差

第 $m$ 轮用弱学习器（如深度为 $J$ 的决策树）拟合伪残差：

$$f_m(x) = \arg\min_f \sum_{i=1}^{n} (r_{im} - f(x_i))^2$$

树将输入空间划分为 $R_{jm}$ 个区域（叶节点），每个区域的输出值：

$$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$$

### 5. 模型更新

$$F_m(x) = F_{m-1}(x) + \nu \cdot \sum_{j=1}^{J} \gamma_{jm} \mathbb{I}[x \in R_{jm}]$$

其中 $\nu$ 是学习率（`learning_rate`），控制每步的步长。

### 6. 学习率与迭代次数的权衡

$$F_M(x) = \sum_{m=1}^{M} \nu \cdot f_m(x)$$

- $\nu$ 小 → 每步贡献小 → 需要更多 $M$ → 训练更慢，但泛化通常更好
- $\nu$ 大 → 每步贡献大 → 收敛快 → 容易过拟合

代码中实验了 $(0.3, 50), (0.1, 100), (0.05, 200), (0.01, 500)$ 的组合。

### 7. 随机梯度提升（Stochastic GB）

引入 `subsample` 参数，每轮只使用 $80\%$ 的随机子集训练：

$$r_{im} = -\frac{\partial L}{\partial F}\bigg|_{F=F_{m-1}}, \quad i \in S_m, \quad |S_m| = n \cdot \text{subsample}$$

类似于 Bagging 的随机性，降低基学习器之间的相关性，减少过拟合。

### 8. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `GradientBoostingClassifier(n_estimators=100)` | 加法模型 $F_{100} = \sum_{m=1}^{100}\alpha_m f_m$ |
| `learning_rate=0.1` | 步长 $\nu=0.1$，控制每轮贡献 |
| `max_depth=3` | 基学习器为深度 $J=3$ 的决策树 |
| `subsample=0.8` | 随机梯度提升，每轮用 $80\%$ 样本 |
| `feature_importances_` | 基于特征在所有树中分裂带来的损失减少之和 |
| `max_depth=1`（弱学习器） | 决策树桩，Boosting 的典型弱学习器 |
