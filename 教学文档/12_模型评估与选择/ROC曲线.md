# ROC 曲线与 AUC：阈值无关的评估

> 所属模块：12_模型评估与选择 | 源文件：ROC曲线.py | 核心功能：ROC 曲线、AUC 计算、多分类 ROC

## 概述

ROC 曲线画出不同阈值下 TPR（真正率）vs FPR（假正率）。AUC（曲线下面积）是一个阈值无关的综合指标。AUC=1 完美，AUC=0.5 随机。

## 关键代码解释

```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
```

## 注意事项

1. 需要概率输出（`predict_proba`），不是类别预测
2. 类别不平衡时 ROC 可能过于乐观，用 PR 曲线
3. 多分类需要 One-vs-Rest 展开

## 延伸思考

- **PR 曲线**：精确率 vs 召回率，不平衡数据更好
- **ROC 的阈值选择**：Youden's J 统计量
- **多分类 AUC**：macro/micro/weighted 平均