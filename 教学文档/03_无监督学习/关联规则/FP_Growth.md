# FP-Growth：比 Apriori 快 100 倍的频繁项集挖掘

> 所属模块：03_无监督学习/关联规则 | 源文件：FP_Growth.py | 核心功能：FP-Tree 构建、无候选集挖掘、与 Apriori 的性能对比

## 概述

Apriori 需要多次扫描数据库并生成大量候选集，在大数据集上效率低下。FP-Growth（Frequent Pattern Growth）算法巧妙地用一棵**前缀树（FP-Tree）**压缩了整个交易数据库，只需扫描两次数据，不需要生成候选集，就能挖掘出所有频繁项集。

脚本在 500 条交易数据上对比了 FP-Growth 和 Apriori 的性能差异，结果通常显示 FP-Growth 快数倍到数十倍。

## 关键代码解释

### 性能对比

`python
for ms in [0.1, 0.15, 0.2]:
    fi_ap = apriori(df, min_support=ms, use_colnames=True)
    fi_fp2 = fpgrowth(df, min_support=ms, use_colnames=True)
`

FP-Growth 和 Apriori 产生**完全相同的结果**，只是计算效率不同。min_support 越低，差距越大。

### 高质量规则过滤

`python
high_quality = rules[(rules["lift"] > 1.5) & (rules["confidence"] > 0.6)]
`

实际应用中通常用多个条件组合过滤，保留真正有价值的规则。

## 使用示例

`python
from mlxtend.frequent_patterns import fpgrowth, association_rules
fi = fpgrowth(df, min_support=0.15, use_colnames=True)
rules = association_rules(fi, metric="lift", min_threshold=1.0)
`

## 注意事项

1. **与 Apriori 结果完全相同**，只是速度更快
2. **FP-Tree 可能占用大量内存**：数据维度很高时注意内存
3. **min_support 仍是关键参数**
4. **适合大规模数据**：小数据集两者差别不大

## 延伸思考

- **FP-Growth 的原理**：第一次扫描统计频率并排序，第二次扫描构建 FP-Tree，然后递归挖掘条件模式基
- **Eclat 算法**：基于等价类的垂直数据格式挖掘，也是 Apriori 的高效替代
- **在线关联规则挖掘**：增量更新 FP-Tree 而不需重建