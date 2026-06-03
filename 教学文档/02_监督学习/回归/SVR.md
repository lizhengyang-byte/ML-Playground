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
4. **概率输出不可用**：SVR 不支持 predict_proba，但可以通过 eturn_std 获得不确定性（需额外计算）

## 延伸思考

- **LinearSVR**：线性核专用实现，比 SVR(kernel="linear") 快得多
- **NuSVR**：用 nu 参数替代 epsilon，控制支持向量比例
- **高斯过程回归**：贝叶斯版本的核回归，自动提供不确定性估计