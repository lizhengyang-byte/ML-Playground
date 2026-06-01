"""
特征重要性 (Feature Importance)

评估每个特征对模型预测的贡献程度，常用方法：
1. 树模型内置的 feature_importances_ - 基于不纯度减少（Gini/信息增益）
2. 排列重要性 (Permutation Importance) - 打乱单个特征后模型性能下降程度
3. 相关性分析 - 特征与目标变量的统计相关系数
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. 树模型内置特征重要性
# ============================================================
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

print("=" * 60)
print("1. 树模型内置特征重要性 (feature_importances_)")
print("=" * 60)

X_cls, y_cls = make_classification(
    n_samples=500, n_features=12, n_informative=5,
    n_redundant=2, n_classes=2, random_state=42
)

feature_names = [f"特征{i}" for i in range(X_cls.shape[1])]

# 随机森林
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_cls, y_cls)

print("随机森林 feature_importances_:")
importance_rf = rf.feature_importances_
for name, imp in sorted(zip(feature_names, importance_rf), key=lambda x: -x[1]):
    bar = "#" * int(imp * 50)
    print(f"  {name}: {imp:.4f} {bar}")

# 梯度提升
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb.fit(X_cls, y_cls)

print("\n梯度提升 feature_importances_:")
importance_gb = gb.feature_importances_
for name, imp in sorted(zip(feature_names, importance_gb), key=lambda x: -x[1]):
    bar = "#" * int(imp * 50)
    print(f"  {name}: {imp:.4f} {bar}")

# ============================================================
# 2. 排列重要性 (Permutation Importance)
# ============================================================
from sklearn.inspection import permutation_importance

print("\n" + "=" * 60)
print("2. 排列重要性 (Permutation Importance)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42
)

rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2.fit(X_train, y_train)

# 排列重要性：打乱每个特征的值，观察准确率下降多少
perm_result = permutation_importance(
    rf2, X_test, y_test, n_repeats=10, random_state=42
)

print("排列重要性 (测试集上，重复10次):")
print(f"  {'特征':<8} {'均值':>8} {'标准差':>8}")
print(f"  {'-'*28}")
for name, mean_imp, std_imp in sorted(
    zip(feature_names, perm_result.importances_mean, perm_result.importances_std),
    key=lambda x: -x[1]
):
    print(f"  {name:<8} {mean_imp:>8.4f} {std_imp:>8.4f}")

# ============================================================
# 3. 相关性分析
# ============================================================
print("\n" + "=" * 60)
print("3. 相关性分析 (Pearson相关系数)")
print("=" * 60)

# 生成回归数据
X_reg, y_reg = make_regression(
    n_samples=300, n_features=8, n_informative=4,
    noise=10, random_state=42
)

# 计算Pearson相关系数
correlations = np.array([
    np.corrcoef(X_reg[:, i], y_reg)[0, 1]
    for i in range(X_reg.shape[1])
])

print("各特征与目标变量的Pearson相关系数:")
print(f"  {'特征':<8} {'相关系数':>10} {'|相关|':>8}")
print(f"  {'-'*30}")
for name, corr in sorted(
    zip([f"特征{i}" for i in range(X_reg.shape[1])], correlations),
    key=lambda x: -abs(x[1])
):
    strength = "强" if abs(corr) > 0.5 else "中" if abs(corr) > 0.3 else "弱"
    print(f"  {name:<8} {corr:>10.4f} {abs(corr):>8.4f} ({strength})")

# ============================================================
# 4. Spearman等级相关 (非线性关系)
# ============================================================
from scipy.stats import spearmanr

print("\n" + "=" * 60)
print("4. Spearman等级相关 (非线性关系)")
print("=" * 60)

# 创建非线性关系: y = x^2 + 噪声
np.random.seed(42)
x_nonlinear = np.random.uniform(-3, 3, 200)
y_nonlinear = x_nonlinear ** 2 + np.random.normal(0, 0.5, 200)

pearson_corr = np.corrcoef(x_nonlinear, y_nonlinear)[0, 1]
spearman_corr, spearman_p = spearmanr(x_nonlinear, y_nonlinear)

print(f"Pearson相关系数 (线性):  {pearson_corr:.4f}")
print(f"Spearman相关系数 (单调): {spearman_corr:.4f}")
print(f"Spearman p值:            {spearman_p:.6f}")
print("说明: 当变量间存在非线性单调关系时，Spearman比Pearson更敏感")

# ============================================================
# 5. 综合对比
# ============================================================
print("\n" + "=" * 60)
print("5. 三种方法的对比")
print("=" * 60)

# 对齐排名
rf_rank = np.argsort(-importance_rf)
perm_rank = np.argsort(-perm_result.importances_mean)
corr_rank = np.argsort(-np.abs(correlations))

print(f"{'特征':<8} {'RF重要性排名':>12} {'排列重要性排名':>14} {'相关系数排名':>12}")
print(f"{'-'*50}")
for i in range(X_cls.shape[1]):
    name = feature_names[i]
    rf_r = np.where(rf_rank == i)[0][0] + 1
    perm_r = np.where(perm_rank == i)[0][0] + 1
    # 树模型和相关性维度不同，这里只展示树模型的
    print(f"  {name:<8} {rf_r:>12} {perm_r:>14}")

print("\n总结:")
print("  - 树模型重要性: 基于训练集，可能偏向高基数特征")
print("  - 排列重要性: 基于测试集，更可靠的泛化评估")
print("  - 相关性分析: 简单快速，但只能捕捉线性/单调关系")
