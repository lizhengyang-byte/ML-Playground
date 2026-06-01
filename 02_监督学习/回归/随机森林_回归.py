"""
随机森林回归 —— Bagging 集成多棵决策树，降低方差提升泛化能力
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ===================== 1. 生成数据 =====================
X, y = make_regression(
    n_samples=500, n_features=15, n_informative=8,
    noise=15, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== 2. 基础随机森林回归 =====================
rf = RandomForestRegressor(
    n_estimators=100, max_features="sqrt",
    oob_score=True, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

print("=== 随机森林回归 ===")
print(f"训练 R²: {rf.score(X_train, y_train):.4f}")
print(f"测试 R²: {rf.score(X_test, y_test):.4f}")
print(f"OOB R²: {rf.oob_score_:.4f}")

y_pred = rf.predict(X_test)
print(f"MSE:  {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")

# ===================== 3. 特征重要性 =====================
print("\n=== 特征重要性 ===")
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
for rank, idx in enumerate(indices, 1):
    bar = "█" * int(importances[idx] * 100)
    print(f"  {rank:>2}. 特征{idx:>2}: {importances[idx]:.4f} {bar}")

# ===================== 4. n_estimators 影响 =====================
print("\n=== n_estimators 对比 ===")
for n in [10, 50, 100, 200, 500]:
    rf_n = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    rf_n.fit(X_train, y_train)
    oob_rf = RandomForestRegressor(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
    oob_rf.fit(X_train, y_train)
    print(f"  n={n:>4}: 测试R²={rf_n.score(X_test, y_test):.4f}, OOB={oob_rf.oob_score_:.4f}")

# ===================== 5. max_features 影响 =====================
print("\n=== max_features 对比 ===")
for mf in ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0]:
    rf_m = RandomForestRegressor(n_estimators=100, max_features=mf, random_state=42, n_jobs=-1)
    rf_m.fit(X_train, y_train)
    test_r2 = rf_m.score(X_test, y_test)
    print(f"  max_features={str(mf):>6}: 测试R²={test_r2:.4f}")

# ===================== 6. 单棵树 vs 随机森林 =====================
print("\n=== 单棵树 vs 随机森林 ===")
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
print(f"决策树:    训练R²={dt.score(X_train, y_train):.4f}, 测试R²={dt.score(X_test, y_test):.4f}")
print(f"随机森林:  训练R²={rf.score(X_train, y_train):.4f}, 测试R²={rf.score(X_test, y_test):.4f}")
print("随机森林通过集成显著降低过拟合（训练R²从1.0降到更合理的值，测试R²大幅提升）")

print("\n=== 随机森林回归要点 ===")
print("- 每棵树用 bootstrap 采样 + 随机特征子集训练")
print("- 最终预测 = 所有树预测的平均值")
print("- OOB 评分提供免费的验证，无需单独划分验证集")
print("- 通常 n_estimators=100~500 即可，更多收益递减")
print("- 不需要特征缩放，能捕捉非线性关系和交互效应")
