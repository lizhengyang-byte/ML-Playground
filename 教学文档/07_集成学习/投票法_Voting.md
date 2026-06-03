# 投票法：最简单的集成策略

> 所属模块：07_集成学习 | 源文件：投票法_Voting.py | 硬投票/软投票、权重调节

## 概述

投票法把多个不同模型的预测结果通过投票（分类）或平均（回归）合并。硬投票取多数类，软投票取概率平均。

## 关键代码解释

```python
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("svm", svm)],
    voting="soft", weights=[1, 2, 1]
)
```

## 注意事项

1. 软投票通常优于硬投票（利用了概率信息）
2. 模型越多样化，集成效果越好
3. 权重可以基于验证集性能手动调节

## 延伸思考

- **加权平均 vs 简单平均**：调权重可能过拟合
- **Blending**：用验证集预测值训练元模型
- **Stacking**：更系统化的多层集成