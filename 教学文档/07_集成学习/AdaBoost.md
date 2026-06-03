# AdaBoost：加权投票的 Boosting 先驱

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