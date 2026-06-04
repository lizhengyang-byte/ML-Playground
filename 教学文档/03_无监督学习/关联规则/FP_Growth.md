# FP-Growth：比 Apriori 快 100 倍的频繁项集挖掘
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：03_无监督学习/关联规则 | 源文件：FP_Growth.py | 核心功能：FP-Tree 构建、无候选集挖掘、与 Apriori 的性能对比

## 概述

Apriori 需要多次扫描数据库并生成大量候选集，在大数据集上效率低下。FP-Growth（Frequent Pattern Growth）算法巧妙地用一棵**前缀树（FP-Tree）**压缩了整个交易数据库，只需扫描两次数据，不需要生成候选集，就能挖掘出所有频繁项集。

脚本在 500 条交易数据上对比了 FP-Growth 和 Apriori 的性能差异，结果通常显示 FP-Growth 快数倍到数十倍。

## 关键代码解释

### 性能对比

```python
for ms in [0.1, 0.15, 0.2]:
    fi_ap = apriori(df, min_support=ms, use_colnames=True)
    fi_fp2 = fpgrowth(df, min_support=ms, use_colnames=True)
```

FP-Growth 和 Apriori 产生**完全相同的结果**，只是计算效率不同。min_support 越低，差距越大。

### 高质量规则过滤

```python
high_quality = rules[(rules["lift"] > 1.5) & (rules["confidence"] > 0.6)]
```

实际应用中通常用多个条件组合过滤，保留真正有价值的规则。

## 使用示例

```python
from mlxtend.frequent_patterns import fpgrowth, association_rules
fi = fpgrowth(df, min_support=0.15, use_colnames=True)
rules = association_rules(fi, metric="lift", min_threshold=1.0)
```

## 注意事项

1. **与 Apriori 结果完全相同**，只是速度更快
2. **FP-Tree 可能占用大量内存**：数据维度很高时注意内存
3. **min_support 仍是关键参数**
4. **适合大规模数据**：小数据集两者差别不大

## 延伸思考

- **FP-Growth 的原理**：第一次扫描统计频率并排序，第二次扫描构建 FP-Tree，然后递归挖掘条件模式基
- **Eclat 算法**：基于等价类的垂直数据格式挖掘，也是 Apriori 的高效替代
- **在线关联规则挖掘**：增量更新 FP-Tree 而不需重建
## 数学原理

### 1. FP-Tree 数据结构

**代码对应**：FP-Growth 通过 FP-Tree 压缩事务数据库，避免多次扫描。

FP-Tree（Frequent Pattern Tree）是一种前缀树结构：
- 每个节点存储一个项及其计数
- 共享相同前缀的事务在树中共享路径
- 头指针表（Header Table）链接每个项的所有节点

**构建过程**：
1. 第一次扫描：统计每个项的支持度，删除非频繁项，按支持度降序排列
2. 第二次扫描：将每个事务按排序后的顺序插入树中，共享前缀路径

### 2. FP-Growth 算法

**核心思想**：将频繁项集挖掘问题分解为多个子问题。

对每个频繁项 $\alpha$：
1. 从 FP-Tree 中提取包含 $\alpha$ 的所有路径（条件模式基）
2. 用条件模式基构建 $\alpha$ 的**条件 FP-Tree**
3. 在条件 FP-Tree 上递归挖掘

### 3. 条件模式基与条件 FP-Tree

项 $\alpha$ 的**条件模式基**（Conditional Pattern Base）：以 $\alpha$ 结尾的所有前缀路径及其计数。

例如，如果项 $e$ 的条件模式基为 $\{(abc: 1), (ac: 2), (b: 1)\}$，则构建 $e$ 的条件 FP-Tree 时，$a$ 出现 3 次，$c$ 出现 3 次，$b$ 出现 2 次。如果 minsup = 2，则 $a$、$c$、$b$ 都进入条件 FP-Tree。

### 4. 与 Apriori 的对比

| 特性 | Apriori | FP-Growth |
|------|---------|-----------|
| 数据结构 | 候选项集 | FP-Tree |
| 数据库扫描次数 | $O(k)$（$k$ 为最大项集大小） | 2 次 |
| 候选生成 | 需要 | 不需要 |
| 时间复杂度 | $O(2^p)$ 最坏 | $O(n \cdot p)$ 通常 |
| 空间复杂度 | $O(p)$ | $O(n \cdot p)$（树存储） |

FP-Growth 通常比 Apriori 快一个数量级，但 FP-Tree 可能很大（极端情况下退化为链表）。

### 5. 支持度、置信度、提升度

FP-Growth 挖掘出的频繁项集同样使用支持度、置信度、提升度评估关联规则（与 Apriori 相同）。

额外的评估指标：

**杠杆率**（Leverage）：$\text{leverage}(X \Rightarrow Y) = P(X \cap Y) - P(X)P(Y)$

**确信度**（Conviction）：$\text{conviction}(X \Rightarrow Y) = \frac{1 - P(Y)}{1 - \text{conf}(X \Rightarrow Y)}$

Conviction 衡量规则预测错误的程度。$\text{conviction} = 1$ 表示 $X$ 和 $Y$ 独立，$\text{conviction} \to \infty$ 表示规则总是正确。
