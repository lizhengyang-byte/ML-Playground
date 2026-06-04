# 投票法：最简单的集成策略
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


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
## 数学原理

### 1. 投票法的基本框架

投票法集成通过聚合多个独立训练的模型的预测来提升性能。设 $K$ 个模型为 $f_1, f_2, \ldots, f_K$，分类问题有 $C$ 个类别。

### 2. 硬投票（Hard Voting / Majority Voting）

最终预测为多数表决：

$$\hat{y} = \arg\max_{c \in \{1,\ldots,C\}} \sum_{k=1}^{K} \mathbb{I}[f_k(x) = c]$$

即选择被最多模型预测的类别。每个模型权重相等，票数相同。

### 3. 软投票（Soft Voting / Weighted Average）

当模型能输出类别概率 $p_k(c|x)$ 时，聚合概率：

$$\hat{y} = \arg\max_c \sum_{k=1}^{K} w_k \cdot p_k(c|x)$$

其中 $w_k$ 是第 $k$ 个模型的权重。代码中 `voting="soft"` 需要模型支持 `predict_proba`。

软投票的优势：考虑了模型的**置信度**。一个模型以 0.99 概率预测类别 A，比以 0.51 概率预测类别 A 的信息量更大。

### 4. 加权投票

引入权重 $w_k$ 反映各模型的可靠程度：

$$\hat{y} = \arg\max_c \sum_{k=1}^{K} w_k \cdot p_k(c|x), \quad w_k \geq 0$$

代码中尝试了不同权重组合：
- `weights=[1,1,1,1,1]`：等权，等价于软投票
- `weights=[2,1,2,2,1]`：对 RF、GB、SVM 给予更高信任
- `weights=[3,1,3,3,1]`：进一步放大强模型的权重

### 5. 投票法有效的条件

投票法提升性能的数学条件（Condorcet 陪审团定理的推广）：

设每个模型独立分类正确的概率为 $p > 0.5$，$K$ 个模型多数投票正确的概率：

$$P(\text{correct}) = \sum_{k=\lceil K/2 \rceil}^{K} \binom{K}{k} p^k (1-p)^{K-k}$$

当 $p > 0.5$ 且 $K \to \infty$ 时，$P(\text{correct}) \to 1$。

关键前提：模型的错误**不相关**。模型多样性越高，投票法效果越好。

### 6. 硬投票 vs 软投票的数学差异

| 方法 | 聚合方式 | 信息利用 | 适用条件 |
|------|----------|----------|----------|
| 硬投票 | 多数表决 $\arg\max \sum \mathbb{I}$ | 仅用类别标签 | 任何分类器 |
| 软投票 | 概率加权平均 $\arg\max \sum w_k p_k$ | 用概率置信度 | 需要 `predict_proba` |

软投票在概率校准良好时通常优于硬投票。

### 7. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `voting="hard"` | $\hat{y} = \arg\max_c \sum \mathbb{I}[f_k(x)=c]$ |
| `voting="soft"` | $\hat{y} = \arg\max_c \sum w_k p_k(c|x)$ |
| `weights=[2,1,2,2,1]` | 加权软投票，$w = (2,1,2,2,1)$ |
| `predict_proba=True` | SVC 输出概率用于软投票 |
| `cross_val_score` | 单模型的 5 折交叉验证准确率 |
| 5 个不同的模型 | $f_1, \ldots, f_5$，多样性是投票法有效的关键 |
