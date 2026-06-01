"""
AdaBoost（自适应提升）—— 经典 Boosting 算法，通过调整样本权重关注难分类样本
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ===================== 1. 生成数据 =====================
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ===================== 2. 基准：决策树桩 =====================
dt_stump = DecisionTreeClassifier(max_depth=1, random_state=42)
dt_stump.fit(X_train, y_train)
print(f"=== 决策树桩: 测试准确率={dt_stump.score(X_test, y_test):.4f} ===")

# ===================== 3. AdaBoost 分类 =====================
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42,
)
ada.fit(X_train, y_train)

print(f"\n=== AdaBoost ===")
print(f"训练准确率: {ada.score(X_train, y_train):.4f}")
print(f"测试准确率: {ada.score(X_test, y_test):.4f}")

# ===================== 4. n_estimators 对比 =====================
print("\n=== n_estimators 对比 ===")
for n in [5, 10, 25, 50, 100, 200]:
    a = AdaBoostClassifier(n_estimators=n, random_state=42)
    a.fit(X_train, y_train)
    print(f"  n={n:>3}: 测试准确率={a.score(X_test, y_test):.4f}")

# ===================== 5. learning_rate 对比 =====================
print("\n=== learning_rate 对比 ===")
for lr in [0.01, 0.1, 0.5, 1.0, 2.0]:
    a = AdaBoostClassifier(n_estimators=50, learning_rate=lr, random_state=42)
    a.fit(X_train, y_train)
    print(f"  lr={lr:>4}: 测试准确率={a.score(X_test, y_test):.4f}")

# ===================== 6. 基学习器复杂度 =====================
print("\n=== 基学习器复杂度影响 ===")
for depth in [1, 2, 3, 5]:
    a = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=depth),
        n_estimators=50, random_state=42,
    )
    a.fit(X_train, y_train)
    print(f"  树深度={depth}: 测试准确率={a.score(X_test, y_test):.4f}")

# ===================== 7. 特征重要性 =====================
print("\n=== 特征重要性 ===")
importances = ada.feature_importances_
for i in np.argsort(importances)[::-1]:
    print(f"  特征{i}: {importances[i]:.4f}")

# ===================== 8. AdaBoost 原理 =====================
print("\n=== AdaBoost 算法原理 ===")
print("1. 初始化样本权重: w_i = 1/n")
print("2. 对每轮 t = 1, ..., T:")
print("   a. 用加权数据训练弱学习器 h_t")
print("   b. 计算加权错误率: ε_t = Σ w_i × I(h_t(x_i) ≠ y_i)")
print("   c. 计算学习器权重: α_t = 0.5 × ln((1-ε_t)/ε_t)")
print("   d. 更新样本权重: w_i ×= exp(±α_t)（错误样本权重增加）")
print("3. 最终: H(x) = sign(Σ α_t × h_t(x))")

# ===================== 9. AdaBoost 回归 =====================
print("\n=== AdaBoost 回归 ===")
from sklearn.datasets import make_regression
X_r, y_r = make_regression(n_samples=300, n_features=10, noise=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
ada_r = AdaBoostRegressor(n_estimators=50, random_state=42)
ada_r.fit(X_tr, y_tr)
print(f"R²: {ada_r.score(X_te, y_te):.4f}")

print("\n=== AdaBoost 要点 ===")
print("- 经典 Boosting：通过样本权重调整关注难分类样本")
print("- 基学习器通常是弱学习器（决策树桩）")
print("- learning_rate: 缩小每轮的贡献，提高泛化")
print("- 对异常值敏感（异常值可能被赋予极大权重）")
print("- 与 GradientBoosting 的区别: AdaBoost 用权重调整，GB 用残差拟合")
