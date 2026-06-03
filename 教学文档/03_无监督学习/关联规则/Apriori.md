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