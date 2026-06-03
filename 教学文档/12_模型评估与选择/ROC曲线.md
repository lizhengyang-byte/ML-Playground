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
﻿## 数学原理

### 1. ROC 曲线的构建

ROC（Receiver Operating Characteristic）曲线以**假正率（FPR）**为横轴、**真正率（TPR）**为纵轴：

$$\text{TPR} = \frac{TP}{TP + FN} = \text{Recall}$$

$$\text{FPR} = \frac{FP}{FP + TN}$$

通过改变分类阈值 $\theta$（从 $1$ 到 $0$），每个 $\theta$ 对应一个 (FPR, TPR) 点，连成曲线。

### 2. AUC（曲线下面积）

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

离散近似：

$$\text{AUC} = \sum_{i=1}^{n-1} \frac{(\text{FPR}_{i+1} - \text{FPR}_i)(\text{TPR}_{i+1} + \text{TPR}_i)}{2}$$

AUC 的概率解释：随机选一个正样本和一个负样本，分类器给正样本打分高于负样本的概率。

$$\text{AUC} = P(\hat{p}(x^+) > \hat{p}(x^-))$$

### 3. AUC 的解读

| AUC 范围 | 模型质量 |
|----------|----------|
| 0.9 - 1.0 | 优秀 |
| 0.8 - 0.9 | 良好 |
| 0.7 - 0.8 | 一般 |
| 0.5 - 0.7 | 较差 |
| 0.5 | 随机猜测（对角线） |

### 4. 多分类 ROC（One-vs-Rest）

对 $C$ 个类别，每个类别 $c$ 构建二分类问题：

$$\text{TPR}_c = \frac{TP_c}{TP_c + FN_c}, \quad \text{FPR}_c = \frac{FP_c}{FP_c + TN_c}$$

宏平均 AUC：$\text{AUC}_{macro} = \frac{1}{C}\sum_{c=1}^{C}\text{AUC}_c$

### 5. Precision-Recall 曲线

当类别**严重不平衡**时（正样本远少于负样本），PR 曲线比 ROC 更有信息量。

$$\text{PR 曲线}: (R, P) = \left(\frac{TP}{TP+FN}, \frac{TP}{TP+FP}\right)$$

**AP（Average Precision）**：PR 曲线下面积的近似

$$\text{AP} = \sum_{k}(R_k - R_{k-1}) P_k$$

### 6. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `roc_curve(y_true, y_score)` | 返回 FPR, TPR, 阈值数组 |
| `auc(fpr, tpr)` | ROC 曲线下面积 |
| `roc_auc_score(y_true, y_score)` | 直接计算 AUC |
| `precision_recall_curve(y_true, y_score)` | 返回 P, R, 阈值 |
| `average_precision_score(y_true, y_score)` | AP 值 |
| `label_binarize(y, classes=...)` | one-hot 编码用于多分类 ROC |
