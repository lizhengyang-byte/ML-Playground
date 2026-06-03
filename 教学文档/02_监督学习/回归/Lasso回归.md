# Lasso 回归：L1 正则化的自动特征选择

> 所属模块：02_监督学习/回归 | 源文件：Lasso回归.py | 核心功能：L1 正则化特征选择、LassoCV 自动选参

## 概述

Lasso（Least Absolute Shrinkage and Selection Operator）回归在损失函数中加入 L1 惩罚项 α||w||₁。与岭回归的关键区别：L1 正则化会把部分系数**精确压缩为 0**，从而实现自动特征选择。

在高维数据中（特征远多于样本，或大部分特征是噪声），Lasso 能自动"筛"出重要特征，这是岭回归做不到的。

脚本构造了一个只有 3 个有用特征的稀疏数据集，演示 Lasso 如何自动识别这些特征，以及 alpha 对稀疏性的影响。

## 关键代码解释

### 特征选择效果

`python
true_coef = np.zeros(p)
true_coef[0] = 5.0; true_coef[3] = -3.0; true_coef[7] = 2.0
# ...
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
np.count_nonzero(lasso.coef_)  # 大约 3-5 个非零系数
`

Lasso 准确地把不重要特征的系数压到了 0，保留了 3 个真正有用的特征。这就是 L1 正则化的"稀疏诱导"特性。

### LassoCV 自动选参

`python
lasso_cv = LassoCV(alphas=None, cv=5, max_iter=10000, random_state=42)
`

lphas=None 让 LassoCV 自动生成一个 alpha 网格，用交叉验证选择最优值。

## 使用示例

`python
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
print(f"最优 alpha: {lasso_cv.alpha_}")
print(f"非零系数: {np.count_nonzero(lasso_cv.coef_)}")
`

## 注意事项

1. **必须特征缩放**：L1 正则化对系数尺度敏感
2. **最多选 min(n, p) 个特征**：Lasso 的选择数量受限于样本数
3. **共线性特征组**：Lasso 倾向于从一组高度相关的特征中只选一个，选择不稳定
4. **max_iter**：Lasso 使用坐标下降法，可能需要更多迭代才能收敛
5. **alpha 越大，非零系数越少**：需要在稀疏性和拟合能力之间找平衡

## 延伸思考

- **Lasso 的几何直觉**：L1 约束区域是菱形，最优解容易落在角点（即某些坐标为 0）
- **Basis Pursuit**：信号处理中的 L1 最小化，与 Lasso 思想相同
- **ElasticNet**：结合 L1 和 L2，解决 Lasso 在共线性组中的选择不稳定性
- **贝叶斯 Lasso**：用拉普拉斯先验替代 L1 罚项，提供不确定性估计