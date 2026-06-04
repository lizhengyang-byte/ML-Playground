"""
随机森林分类 —— Bagging 思想 + 随机特征选择，集成多棵决策树提升泛化能力
"""
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ===================== 1. 构造数据 =====================
X, y = make_classification(
    n_samples=500, n_features=15, n_informative=8,
    n_classes=3, n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 2. 基础随机森林 =====================
# n_estimators: 树的数量，越多越稳定但计算量越大
# max_features: 每棵树随机选择的特征数（分类默认 sqrt(n_features)）
rf = RandomForestClassifier(
    n_estimators=100, max_features="sqrt",
    oob_score=True, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

print("=== 随机森林分类 ===")
print(f"训练集准确率: {rf.score(X_train, y_train):.4f}")
print(f"测试集准确率: {rf.score(X_test, y_test):.4f}")
print(f"OOB（袋外）准确率: {rf.oob_score_:.4f}")
# OOB 评分：每棵树用未参与训练的样本评估，无需单独验证集

# ===================== 3. 特征重要性 =====================
print(f"\n=== 特征重要性（前 10）===")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
for rank, idx in enumerate(indices, 1):
    print(f"  {rank}. 特征{idx}: {importances[idx]:.4f}")

# ===================== 4. 不同参数对比 =====================
print("\n=== n_estimators 对比 ===")
for n in [10, 50, 100, 200, 500]:
    rf_n = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf_n.fit(X_train, y_train)
    test_acc = rf_n.score(X_test, y_test)
    oob = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
    oob.fit(X_train, y_train)
    print(f"  n={n:>4}: 测试准确率={test_acc:.4f}, OOB={oob.oob_score_:.4f}")

# ===================== 5. 袋外样本评估 =====================
# oob_score=True 时，每棵树对未抽到的样本做预测，取投票结果
print("\n=== 袋外（OOB）评估原理 ===")
print(f"每棵树约使用 {int(0.632 * len(X_train))} 个样本训练（bootstrap 约 63.2%）")
print(f"剩余 ~{len(X_train) - int(0.632 * len(X_train))} 个样本作为该树的 OOB 样本")
print(f"OOB 评分等价于交叉验证，无需额外划分验证集")

# ===================== 6. 预测与分类报告 =====================
y_pred = rf.predict(X_test)
print(f"\n=== 分类报告 ===")
print(classification_report(y_test, y_pred))

# ===================== 7. 随机性来源 =====================
print("\n=== 随机森林的两重随机性 ===")
print("1. 样本随机：每棵树通过 bootstrap 有放回抽样获得不同训练子集")
print("2. 特征随机：每次分裂只考虑随机选取的 max_features 个特征")
print("两重随机使各棵树多样化，降低过拟合风险")
