# SVR：支持向量回归——用 ε-管道拟合

> 所属模块：02_监督学习/回归 | 源文件：SVR.py | 核心功能：ε-不敏感损失、C/epsilon/gamma 参数分析

## 概述

SVR（Support Vector Regression）是 SVM 的回归版本。与传统回归不同，SVR 使用 **ε-不敏感损失**：只要预测误差在 ε 范围内，就不计入损失。这形成了一个围绕回归线的"管道"——管道内的样本不算误差，只有落在管道外的样本（支持向量）才贡献损失。

脚本对比了 4 种核函数，并深入分析了 C、epsilon 和 gamma 三个参数的影响。

## 关键代码解释

### 目标变量也需要缩放

`python
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
`

SVR 基于距离计算，不仅特征需要缩放，**目标变量也需要缩放**。这与大多数回归模型不同。

### epsilon 参数——管道宽度

`python
for eps in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
    svr_e = SVR(kernel="rbf", C=10.0, epsilon=eps)
`

epsilon 越大，管道越宽，支持向量越少，模型越简单。epsilon=0 退化为最小绝对偏差回归。

## 使用示例

`python
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1))
])
pipe.fit(X_train, y_train)
`

## 注意事项

1. **特征和目标都要缩放**
2. **训练慢**：O(n²~n³)，大数据集用 LinearSVR 或 SGDRegressor
3. **C 和 epsilon 需联合调参**
4. **概率输出不可用**：SVR 不支持 predict_proba，但可以通过 
eturn_std 获得不确定性（需额外计算）

## 延伸思考

- **LinearSVR**：线性核专用实现，比 SVR(kernel="linear") 快得多
- **NuSVR**：用 nu 参数替代 epsilon，控制支持向量比例
- **高斯过程回归**：贝叶斯版本的核回归，自动提供不确定性估计
﻿## 数学原理

### 1. $\varepsilon$-不敏感损失函数

**代码对应**：`SVR(kernel="rbf", C=1.0, epsilon=0.1)` 中 epsilon 参数控制管道宽度。

SVR 的核心创新是 **$\varepsilon$-不敏感损失**（$\varepsilon$-insensitive loss）：

$$L_\varepsilon(y, f(\mathbf{x})) = \begin{cases} 0 & \text{if } |y - f(\mathbf{x})| \leq \varepsilon \\ |y - f(\mathbf{x})| - \varepsilon & \text{otherwise} \end{cases}$$

预测误差在 $\varepsilon$ 管道内不计入损失。与平方损失的对比：

| 损失函数 | 公式 | 对异常值敏感度 |
|----------|------|-------------|
| 平方损失 | $(y - \hat{y})^2$ | 高（二次放大） |
| 绝对损失 | $|y - \hat{y}|$ | 中 |
| $\varepsilon$-不敏感 | $\max(0, |y - \hat{y}| - \varepsilon)$ | 低（管道内零损失） |

### 2. SVR 的优化问题

**原始形式**：

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}\|\mathbf{w}\|_2^2 + C\sum_{i=1}^{n}(\xi_i + \xi_i^*)$$

$$\text{s.t.} \quad y_i - \mathbf{w}^T\phi(\mathbf{x}_i) - b \leq \varepsilon + \xi_i$$

$$\mathbf{w}^T\phi(\mathbf{x}_i) + b - y_i \leq \varepsilon + \xi_i^*$$

$$\xi_i, \xi_i^* \geq 0$$

其中 $\xi_i, \xi_i^*$ 为松弛变量，$C$ 控制对超出管道的样本的惩罚强度。

### 3. 对偶问题与核技巧

引入拉格朗日乘子 $\alpha_i, \alpha_i^*$，对偶问题为：

$$\max_{\boldsymbol{\alpha}, \boldsymbol{\alpha}^*} -\frac{1}{2}\sum_{i,j}(\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)K(\mathbf{x}_i, \mathbf{x}_j) - \varepsilon\sum_{i}(\alpha_i + \alpha_i^*) + \sum_{i} y_i(\alpha_i - \alpha_i^*)$$

$$\text{s.t.} \quad 0 \leq \alpha_i, \alpha_i^* \leq C, \quad \sum_{i}(\alpha_i - \alpha_i^*) = 0$$

预测函数为：

$$f(\mathbf{x}) = \sum_{i=1}^{n}(\alpha_i - \alpha_i^*)K(\mathbf{x}_i, \mathbf{x}) + b$$

只有 $\alpha_i - \alpha_i^* \neq 0$ 的样本是**支持向量**。

**代码对应**：代码中 `svr.support_.shape[0]` 返回支持向量数。支持向量越少，模型越稀疏。

### 4. C 和 $\varepsilon$ 的几何意义

**代码对应**：代码中分别对比了不同 C 和 epsilon 的效果。

- **$C$ 越大**：对超出管道的惩罚越重，管道越"硬"，支持向量越多，模型越复杂
- **$\varepsilon$ 越大**：管道越宽，更多样本落在管道内（零损失），支持向量越少，模型越简单

最优 $C$ 和 $\varepsilon$ 通常通过交叉验证选择。实践中，先固定 $\varepsilon$（如 0.1），搜索 $C$，再微调 $\varepsilon$。

### 5. 核函数选择

**代码对应**：代码对比了 linear、rbf、poly、sigmoid 四种核。

RBF 核（高斯核）最常用：

$$K(\mathbf{x}, \mathbf{z}) = \exp\left(-\gamma\|\mathbf{x} - \mathbf{z}\|_2^2\right)$$

$\gamma$ 控制单个样本的影响范围。$\gamma$ 越大，核函数越"窄"，决策边界越弯曲（过拟合风险）。

**代码对应**：`for gamma in ["scale", "auto", 0.01, ...]` 展示了 gamma 的影响。sklearn 默认 `gamma="scale"` 即 $\gamma = 1/(p \cdot \text{Var}(X))$。

### 6. 训练复杂度

SVR 的训练需要求解二次规划问题，复杂度为：

- 约 $O(n^2 \cdot p)$ 到 $O(n^3)$，取决于实现
- 对大数据集（$n > 10^4$），考虑 `LinearSVR` 或 SGDRegressor 替代
