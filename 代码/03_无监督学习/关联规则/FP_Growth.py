"""
FP-Growth 算法 —— 比 Apriori 更高效的频繁项集挖掘，不需要生成候选集
需要安装: pip install mlxtend
"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder
import time

# ===================== 1. 构造较大规模交易数据 =====================
np.random.seed(42)
items = ["牛奶", "面包", "黄油", "鸡蛋", "尿布", "啤酒", "可乐", "薯片", "巧克力", "苹果"]
n_transactions = 500

transactions = []
for _ in range(n_transactions):
    # 随机选择 2~5 个商品
    n_items = np.random.randint(2, 6)
    basket = list(np.random.choice(items, n_items, replace=False))
    # 添加一些关联模式
    if np.random.rand() < 0.4:
        basket.extend(["牛奶", "面包"])
    if np.random.rand() < 0.3:
        basket.extend(["尿布", "啤酒"])
    basket = list(set(basket))  # 去重
    transactions.append(basket)

print(f"交易总数: {len(transactions)}")
print(f"平均每笔交易商品数: {np.mean([len(t) for t in transactions]):.1f}")

# ===================== 2. 数据编码 =====================
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print(f"交易矩阵形状: {df.shape}")

# ===================== 3. FP-Growth 挖掘 =====================
print("\n=== FP-Growth (min_support=0.15) ===")
t0 = time.time()
fi_fp = fpgrowth(df, min_support=0.15, use_colnames=True)
t_fp = time.time() - t0
print(f"耗时: {t_fp:.4f}s")
print(f"频繁项集数: {len(fi_fp)}")
print(fi_fp.sort_values("support", ascending=False).head(10).to_string())

# ===================== 4. 与 Apriori 对比 =====================
print("\n=== FP-Growth vs Apriori 性能对比 ===")
for ms in [0.1, 0.15, 0.2]:
    t0 = time.time()
    fi_ap = apriori(df, min_support=ms, use_colnames=True)
    t_ap = time.time() - t0

    t0 = time.time()
    fi_fp2 = fpgrowth(df, min_support=ms, use_colnames=True)
    t_fp2 = time.time() - t0

    print(f"  min_support={ms}: Apriori={t_ap:.4f}s ({len(fi_ap)} 项集), "
          f"FP-Growth={t_fp2:.4f}s ({len(fi_fp2)} 项集)")

# ===================== 5. 生成关联规则 =====================
rules = association_rules(fi_fp, metric="confidence", min_threshold=0.5)
print(f"\n=== 关联规则 (confidence≥0.5) ===")
print(f"共 {len(rules)} 条规则")

# 按 lift 排序
rules_lift = rules.sort_values("lift", ascending=False)
for _, row in rules_lift.head(8).iterrows():
    ant = ", ".join(row["antecedents"])
    con = ", ".join(row["consequents"])
    print(f"  {ant} → {con}")
    print(f"    S={row['support']:.3f}, C={row['confidence']:.3f}, L={row['lift']:.3f}")

# ===================== 6. 提升度 vs 置信度过滤 =====================
print("\n=== 高提升度高置信度规则 ===")
high_quality = rules[(rules["lift"] > 1.5) & (rules["confidence"] > 0.6)]
print(f"满足条件 (lift>1.5 & confidence>0.6) 的规则: {len(high_quality)} 条")
for _, row in high_quality.iterrows():
    ant = ", ".join(row["antecedents"])
    con = ", ".join(row["consequents"])
    print(f"  {ant} → {con} (L={row['lift']:.3f}, C={row['confidence']:.3f})")

# ===================== 7. 算法原理 =====================
print("\n=== FP-Growth 算法原理 ===")
print("1. 第一次扫描：统计每个项的频率，按频率降序排序")
print("2. 第二次扫描：构建 FP-Tree（前缀树）")
print("   - 每条交易按频率排序后插入树中")
print("   - 共享公共前缀，节省空间")
print("3. 递归挖掘：从每个频繁项出发，提取条件模式基")
print("   - 构建条件 FP-Tree")
print("   - 在条件树上递归挖掘")

print("\n=== FP-Growth vs Apriori ===")
print("Apriori: 需要多次扫描数据 + 生成候选集 → 慢")
print("FP-Growth: 只扫描 2 次 + 不需要候选集 → 快")
print("FP-Growth 在大数据集上可快 10~100 倍")

print("\n=== FP-Growth 要点 ===")
print("- 优点：不需要生成候选集，只需扫描数据库两次，速度快")
print("- 缺点：FP-Tree 可能很大，内存消耗高")
print("- min_support 是最关键的参数")
print("- 与 Apriori 产生完全相同的结果")
