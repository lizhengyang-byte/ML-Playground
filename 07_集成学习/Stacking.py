"""
Stacking —— 堆叠集成：用一个元学习器组合多个基学习器的预测结果
"""
import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ===================== 1. 加载数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ===================== 2. 定义基学习器 =====================
base_estimators = [
    ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
]

# ===================== 3. 手动实现 Stacking =====================
print("=== 手动 Stacking ===")
# 第一层：用交叉预测生成元特征
meta_features_train = []
for name, model in base_estimators:
    cv_pred = cross_val_predict(model, X_train, y_train, cv=5, method="predict")
    # 也可用 predict_proba 获得概率特征
    meta_features_train.append(cv_pred)
meta_X_train = np.column_stack(meta_features_train)

# 第二层：用元学习器训练
meta_model = LogisticRegression(random_state=42)
meta_model.fit(meta_X_train, y_train)

# 在测试集上预测
meta_features_test = []
for name, model in base_estimators:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    meta_features_test.append(pred)
meta_X_test = np.column_stack(meta_features_test)
y_pred = meta_model.predict(meta_X_test)

print(f"Stacking 准确率: {accuracy_score(y_test, y_pred):.4f}")

# 各基学习器单独准确率
print("\n各基学习器单独表现:")
for name, model in base_estimators:
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"  {name:>5}: {acc:.4f}")

# ===================== 4. StackingClassifier（sklearn 封装）=====================
print("\n=== StackingClassifier (sklearn) ===")
stack_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=False,  # True: 元学习器也接收原始特征
)
stack_clf.fit(X_train, y_train)
print(f"训练准确率: {stack_clf.score(X_train, y_train):.4f}")
print(f"测试准确率: {stack_clf.score(X_test, y_test):.4f}")

# ===================== 5. passthrough 模式 =====================
print("\n=== passthrough=True（原始特征 + 基学习器输出）===")
stack_pt = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=True,
)
stack_pt.fit(X_train, y_train)
print(f"测试准确率: {stack_pt.score(X_test, y_test):.4f}")

# ===================== 6. 不同元学习器 =====================
print("\n=== 不同元学习器 ===")
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVCC

for name, final in [("LogisticRegression", LogisticRegression(max_iter=1000)),
                     ("RandomForest", RFC(n_estimators=50, random_state=42)),
                     ("SVC", SVCC(kernel="rbf", probability=True))]:
    s = StackingClassifier(estimators=base_estimators, final_estimator=final, cv=5)
    s.fit(X_train, y_train)
    print(f"  {name:>20}: 测试准确率={s.score(X_test, y_test):.4f}")

# ===================== 7. Stacking 回归 =====================
print("\n=== Stacking 回归 ===")
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso

X_r, y_r = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

base_reg = [
    ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),
    ("gb", GradientBoostingRegressor(n_estimators=50, random_state=42)),
    ("ridge", Ridge()),
]
stack_reg = StackingRegressor(estimators=base_reg, final_estimator=Ridge(), cv=5)
stack_reg.fit(X_tr, y_tr)
print(f"Stacking 回归 R²: {stack_reg.score(X_te, y_te):.4f}")

# ===================== 8. Stacking 要点 =====================
print("\n=== Stacking 要点 ===")
print("- 第一层: 多个不同的基学习器（多样化是关键）")
print("- 第二层: 元学习器学习如何组合基学习器的输出")
print("- 使用交叉预测生成元特征（防止信息泄露）")
print("- passthrough=True: 元学习器同时看到原始特征和基学习器输出")
print("- 基学习器应尽量多样化（不同算法/不同参数）")
print("- 常用于竞赛中提升最终成绩（0.1~0.5% 的提升）")
