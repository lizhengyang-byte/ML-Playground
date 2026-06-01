"""
Boosting —— 串行集成方法，每轮关注前一轮的错误样本，逐步提升
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

# ===================== 1. 生成数据 =====================
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ===================== 2. 基准：单个弱学习器 =====================
dt = DecisionTreeClassifier(max_depth=1, random_state=42)  # 决策树桩
dt.fit(X_train, y_train)
print(f"=== 单个决策树桩: 测试准确率={dt.score(X_test, y_test):.4f} ===")

# ===================== 3. GradientBoosting 分类 =====================
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,  # 随机梯度提升（Stochastic GB）
    random_state=42,
)
gb.fit(X_train, y_train)

print(f"\n=== GradientBoosting ===")
print(f"训练准确率: {gb.score(X_train, y_train):.4f}")
print(f"测试准确率: {gb.score(X_test, y_test):.4f}")
print(f"特征重要性: {gb.feature_importances_.round(4)}")

# ===================== 4. n_estimators 影响 =====================
print("\n=== n_estimators 对比 ===")
for n in [10, 25, 50, 100, 200, 500]:
    gb_n = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1, random_state=42)
    gb_n.fit(X_train, y_train)
    print(f"  n={n:>3}: 测试准确率={gb_n.score(X_test, y_test):.4f}")

# ===================== 5. learning_rate vs n_estimators =====================
print("\n=== learning_rate vs n_estimators ===")
print("learning_rate 越小，需要更多轮次，但泛化通常更好")
for lr, n in [(0.3, 50), (0.1, 100), (0.05, 200), (0.01, 500)]:
    gb_lr = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, random_state=42)
    gb_lr.fit(X_train, y_train)
    print(f"  lr={lr}, n={n:>3}: 测试准确率={gb_lr.score(X_test, y_test):.4f}")

# ===================== 6. subsample（随机梯度提升）=====================
print("\n=== subsample 对比 ===")
for ss in [0.5, 0.7, 0.8, 1.0]:
    gb_s = GradientBoostingClassifier(n_estimators=100, subsample=ss, random_state=42)
    gb_s.fit(X_train, y_train)
    print(f"  subsample={ss}: 测试准确率={gb_s.score(X_test, y_test):.4f}")

# ===================== 7. Boosting 回归 =====================
print("\n=== GradientBoosting 回归 ===")
from sklearn.datasets import make_regression
X_r, y_r = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
gb_r = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_r.fit(X_tr, y_tr)
print(f"R²: {gb_r.score(X_te, y_te):.4f}")

# ===================== 8. Boosting 原理 =====================
print("\n=== Boosting 原理 ===")
print("1. 初始化弱学习器 F_0(x)")
print("2. 对每轮 m = 1, 2, ..., M:")
print("   a. 计算残差（负梯度）r_m = -∂L/∂F")
print("   b. 训练新弱学习器拟合残差")
print("   c. 更新 F_m = F_{m-1} + η × h_m(x)")
print("3. 最终模型: F_M(x) = Σ η × h_m(x)")

print("\n=== Boosting vs Bagging ===")
print("Bagging: 并行训练，降低方差，适合不稳定模型")
print("Boosting: 串行训练，降低偏差，适合弱学习器")
print("Boosting 更容易过拟合（需限制 n_estimators/learning_rate）")

print("\n=== Boosting 要点 ===")
print("- learning_rate × n_estimators 的权衡：小 lr + 大 n 通常更好")
print("- max_depth 通常较小（3~6），基学习器是弱学习器")
print("- subsample 引入随机性，防止过拟合")
print("- 对异常值敏感（残差可能被异常值主导）")
