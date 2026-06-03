# Apriori 算法：从购物车中发现隐藏的关联

> 所属模块：03_无监督学习/关联规则 | 源文件：Apriori.py | 核心功能：频繁项集挖掘、关联规则生成、支持度/置信度/提升度分析

## 概述

"买尿布的人往往也会买啤酒"——这是数据挖掘领域最经典的段子，背后用到的就是关联规则挖掘。Apriori 算法是最经典的关联规则算法，它的核心洞察非常精巧：**如果一个项集是频繁的，那么它的所有子集也必须是频繁的**（反单调性）。利用这个性质，可以大幅剪枝，减少候选集的搜索空间。

脚本模拟了超市购物数据，演示了 Apriori 挖掘频繁项集、生成关联规则，以及支持度/置信度/提升度三个核心指标的解读。

## 代码结构

| 段落 | 内容 |
|------|------|
| 交易数据 | 15 条模拟购物记录 |
| 数据编码 | TransactionEncoder 将交易列表转为 0/1 矩阵 |
| 频繁项集 | priori() 按 min_support 挖掘 |
| 关联规则 | ssociation_rules() 从频繁项集生成规则 |
| 指标排序 | 按 lift 排序展示最有价值的规则 |

## 关键代码解释

### 数据编码

`python
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
`

TransactionEncoder 把交易列表（每条交易是一个商品列表）转为 one-hot 矩阵：每列是一个商品，每行是一条交易，值为 True/False。这是 Apriori 算法的标准输入格式。

### 三个核心指标

`python
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
`

- **支持度**（Support）：P(A∩B)，A 和 B 同时出现的概率——规则的"普遍性"
- **置信度**（Confidence）：P(B|A)，买了 A 也买了 B 的概率——规则的"可靠性"
- **提升度**（Lift）：P(B|A) / P(B)，A 对 B 的实际促进倍数——规则的"价值"

> **Lift > 1 才有意义**。Lift = 2 意味着买了 A 的人买 B 的概率是普通人的 2 倍。

## 使用示例

`python
from mlxtend.frequent_patterns import apriori, association_rules
frequent = apriori(df, min_support=0.2, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=1.0)
`

## 注意事项

1. **min_support 太低** → 候选集爆炸，计算量指数级增长
2. **min_support 太高** → 丢失低频但有价值的规则
3. **提升度才是最有价值的指标**：高置信度可能只是因为 B 本身就很常见
4. **需要 mlxtend 库**：pip install mlxtend
5. **大数据集效率低**：考虑 FP-Growth 替代

## 延伸思考

- **闭频繁项集**和**最大频繁项集**：压缩频繁项集的表示
- **多层关联规则**：在商品分类体系的不同层级挖掘规则
- **时序关联规则**：考虑购买的先后顺序（先买 A 再买 B）
﻿## 数学原理

### 1. 关联规则的基本概念

**代码对应**：Apriori 算法从事务数据中挖掘频繁项集和关联规则。

设事务数据库 $D = \{T_1, T_2, \ldots, T_N\}$，每个事务 $T_i$ 是项目的集合。

**支持度**（Support）：项集 $X$ 在所有事务中出现的频率：

$$\text{supp}(X) = \frac{|\{T \in D : X \subseteq T\}|}{|D|}$$

**置信度**（Confidence）：规则 $X \Rightarrow Y$ 的条件概率：

$$\text{conf}(X \Rightarrow Y) = P(Y|X) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)}$$

**提升度**（Lift）：规则的"有趣程度"：

$$\text{lift}(X \Rightarrow Y) = \frac{\text{conf}(X \Rightarrow Y)}{\text{supp}(Y)} = \frac{P(X \cap Y)}{P(X)P(Y)}$$

- $\text{lift} = 1$：$X$ 和 $Y$ 独立
- $\text{lift} > 1$：正相关
- $\text{lift} < 1$：负相关

### 2. Apriori 性质（先验性质）

**向下闭包性**（Downward Closure）：如果项集 $X$ 是频繁的（$\text{supp}(X) \geq \text{min\_sup}$），则 $X$ 的所有子集也是频繁的。

逆否命题：如果项集 $X$ 是非频繁的，则 $X$ 的所有超集也是非频繁的。

**代码对应**：Apriori 算法利用此性质**剪枝**——一旦发现某个项集非频繁，就不再扩展它，大幅减少搜索空间。

### 3. Apriori 算法流程

1. **扫描 1**：找出所有频繁 1-项集 $L_1$
2. **连接**：由 $L_k$ 生成候选 $(k+1)$-项集 $C_{k+1}$
3. **剪枝**：删除包含非频繁子集的候选
4. **扫描**：计算 $C_{k+1}$ 中每个候选的支持度
5. **保留**：支持度 $\geq \text{min\_sup}$ 的候选进入 $L_{k+1}$
6. 重复直到 $L_{k+1} = \emptyset$

### 4. 关联规则生成

对每个频繁项集 $X$，生成所有非空子集 $Y \subset X$，如果 $\text{conf}(Y \Rightarrow X \setminus Y) \geq \text{min\_conf}$，则输出规则。

### 5. 计算复杂度

Apriori 的瓶颈是**多次数据库扫描**（每轮候选生成后都要扫描数据库计算支持度）。对 $p$ 个不同项目，最坏情况需扫描 $O(2^p)$ 个候选。

FP-Growth 通过 FP-Tree 数据结构避免了候选生成和多次扫描，效率显著更高。
