"""
Apriori 算法 —— 经典关联规则挖掘算法，基于"频繁项集的子集必然是频繁的"剪枝
需要安装: pip install mlxtend
"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ===================== 1. 构造交易数据 =====================
# 模拟超市购物记录
transactions = [
    ["牛奶", "面包", "黄油"],
    ["面包", "尿布", "啤酒", "鸡蛋"],
    ["牛奶", "尿布", "啤酒", "可乐"],
    ["面包", "牛奶", "尿布", "啤酒"],
    ["面包", "牛奶", "尿布", "可乐"],
    ["面包", "牛奶", "啤酒"],
    ["牛奶", "尿布", "啤酒"],
    ["面包", "鸡蛋", "黄油"],
    ["面包", "牛奶", "尿布", "啤酒", "鸡蛋"],
    ["牛奶", "啤酒", "可乐"],
    ["面包", "牛奶", "尿布"],
    ["面包", "牛奶", "啤酒"],
    ["尿布", "啤酒", "可乐"],
    ["面包", "牛奶", "尿布", "啤酒"],
    ["牛奶", "鸡蛋", "黄油"],
]

print(f"交易总数: {len(transactions)}")
print(f"所有商品: {sorted(set(item for t in transactions for item in t))}")

# ===================== 2. 数据编码 =====================
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(f"\n交易矩阵形状: {df.shape}")
print(df.head())

# ===================== 3. Apriori 挖掘频繁项集 =====================
# min_support: 最小支持度阈值
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
print(f"\n=== 频繁项集 (min_support=0.2) ===")
print(f"共找到 {len(frequent_itemsets)} 个频繁项集")
print(frequent_itemsets.sort_values("support", ascending=False).head(15).to_string())

# ===================== 4. 生成关联规则 =====================
# min_threshold: 最小置信度
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(f"\n=== 关联规则 (confidence≥0.6) ===")
print(f"共 {len(rules)} 条规则")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].to_string())

# ===================== 5. 按提升度排序 =====================
print("\n=== 按 Lift 排序（前 10）===")
rules_sorted = rules.sort_values("lift", ascending=False)
for _, row in rules_sorted.head(10).iterrows():
    ant = ", ".join(row["antecedents"])
    con = ", ".join(row["consequents"])
    print(f"  {ant} → {con}")
    print(f"    支持度={row['support']:.3f}, 置信度={row['confidence']:.3f}, "
          f"提升度={row['lift']:.3f}")

# ===================== 6. 不同参数对比 =====================
print("\n=== 不同 min_support 对比 ===")
for ms in [0.1, 0.2, 0.3, 0.4]:
    fi = apriori(df, min_support=ms, use_colnames=True)
    print(f"  min_support={ms}: 频繁项集数={len(fi)}")

# ===================== 7. 指标解释 =====================
print("\n=== 关联规则指标解释 ===")
print("支持度 (Support): P(A∩B) — 交易中同时包含 A 和 B 的比例")
print("置信度 (Confidence): P(B|A) — 包含 A 的交易中也包含 B 的比例")
print("提升度 (Lift): P(B|A)/P(B) — A 的出现对 B 出现概率的提升倍数")
print("  Lift > 1: 正相关（A 促进 B）")
print("  Lift = 1: 独立")
print("  Lift < 1: 负相关（A 抑制 B）")

print("\n=== Apriori 要点 ===")
print("- 剪枝策略：频繁项集的子集必然是频繁的（反单调性）")
print("- min_support 太低 → 计算量爆炸；太高 → 丢失有意义的规则")
print("- min_confidence 太低 → 规则太多且质量差")
print("- Lift 是最有价值的指标，衡量实际关联强度")
print("- 对长事务和高维数据效率较低（考虑 FP-Growth）")
