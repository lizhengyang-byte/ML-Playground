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