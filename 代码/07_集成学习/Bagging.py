"""
Bagging（自助聚合）—— 通过有放回采样训练多个基学习器，投票/平均聚合结果
"""
import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

# ===================== 1. 生成数据 =====================
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ===================== 2. 单个决策树（基准）=====================
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("=== 单个决策树 ===")
print(f"测试准确率: {dt.score(X_test, y_test):.4f}")

# ===================== 3. Bagging 集成 =====================
# n_estimators: 基学习器数量
# max_samples: 每个基学习器使用的样本比例
# max_features: 每个基学习器使用的特征比例
# bootstrap: 是否有放回采样（True=Bagging, False=Pasting）
# bootstrap_features: 是否对特征也做有放回采样

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
bag.fit(X_train, y_train)

print(f"\n=== Bagging 集成 ===")
print(f"基学习器数量: {bag.n_estimators}")
print(f"训练准确率: {bag.score(X_train, y_train):.4f}")
print(f"测试准确率: {bag.score(X_test, y_test):.4f}")

# ===================== 4. OOB 评分 =====================
bag_oob = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    oob_score=True,
    random_state=42,
)
bag_oob.fit(X_train, y_train)
print(f"OOB 准确率: {bag_oob.oob_score_:.4f}")

# ===================== 5. 参数对比 =====================
print("\n=== n_estimators 对比 ===")
for n in [5, 10, 25, 50, 100, 200]:
    b = BaggingClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    b.fit(X_train, y_train)
    print(f"  n={n:>3}: 测试准确率={b.score(X_test, y_test):.4f}")

print("\n=== max_samples 对比 ===")
for ms in [0.3, 0.5, 0.7, 0.8, 1.0]:
    b = BaggingClassifier(n_estimators=50, max_samples=ms, random_state=42)
    b.fit(X_train, y_train)
    print(f"  max_samples={ms}: 测试准确率={b.score(X_test, y_test):.4f}")

# ===================== 6. bootstrap vs pasting =====================
print("\n=== bootstrap vs pasting ===")
for bootstrap in [True, False]:
    b = BaggingClassifier(n_estimators=50, bootstrap=bootstrap, random_state=42)
    b.fit(X_train, y_train)
    name = "Bagging" if bootstrap else "Pasting"
    print(f"  {name:>10}: 测试准确率={b.score(X_test, y_test):.4f}")

# ===================== 7. 特征子采样 =====================
print("\n=== 特征子采样 ===")
for mf in [0.3, 0.5, 0.7, 1.0]:
    b = BaggingClassifier(n_estimators=50, max_features=mf, random_state=42)
    b.fit(X_train, y_train)
    print(f"  max_features={mf}: 测试准确率={b.score(X_test, y_test):.4f}")

# ===================== 8. Bagging 回归 =====================
print("\n=== Bagging 回归 ===")
from sklearn.datasets import make_regression
X_r, y_r = make_regression(n_samples=300, n_features=10, noise=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

dt_r = DecisionTreeRegressor(random_state=42).fit(X_tr, y_tr)
bag_r = BaggingRegressor(n_estimators=50, random_state=42).fit(X_tr, y_tr)
print(f"决策树 R²: {dt_r.score(X_te, y_te):.4f}")
print(f"Bagging R²: {bag_r.score(X_te, y_te):.4f}")

print("\n=== Bagging 要点 ===")
print("- 核心：有放回采样 + 独立训练 + 投票/平均")
print("- 降低方差（适合高方差低偏差的基学习器如决策树）")
print("- OOB 评分：每个基学习器约用 63.2% 数据，剩余 36.8% 作 OOB 评估")
print("- 与随机森林的区别：随机森林额外做了特征随机选择")
