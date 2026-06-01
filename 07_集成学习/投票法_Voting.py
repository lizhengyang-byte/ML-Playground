"""
投票法集成（Voting）—— 多个模型独立预测，通过投票（分类）或平均（回归）聚合
"""
import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ===================== 1. 加载数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ===================== 2. 定义各模型 =====================
models = [
    ("lr", LogisticRegression(max_iter=200, random_state=42)),
    ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
]

# ===================== 3. 各模型单独表现 =====================
print("=== 各模型单独表现 ===")
for name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  {name:>5}: CV={scores.mean():.4f}±{scores.std():.4f}, Test={test_acc:.4f}")

# ===================== 4. 硬投票（多数表决）=====================
print("\n=== 硬投票 (hard voting) ===")
voting_hard = VotingClassifier(estimators=models, voting="hard", n_jobs=-1)
voting_hard.fit(X_train, y_train)
print(f"测试准确率: {voting_hard.score(X_test, y_test):.4f}")

# ===================== 5. 软投票（概率平均）=====================
print("\n=== 软投票 (soft voting) ===")
voting_soft = VotingClassifier(estimators=models, voting="soft", n_jobs=-1)
voting_soft.fit(X_train, y_train)
print(f"测试准确率: {voting_soft.score(X_test, y_test):.4f}")

# ===================== 6. 加权投票 =====================
print("\n=== 加权投票 ===")
# 给不同模型不同权重
for weights in [[1,1,1,1,1], [2,1,2,2,1], [3,1,3,3,1]]:
    vw = VotingClassifier(estimators=models, voting="soft", weights=weights, n_jobs=-1)
    vw.fit(X_train, y_train)
    print(f"  weights={weights}: 测试准确率={vw.score(X_test, y_test):.4f}")

# ===================== 7. 预测概率 =====================
print("\n=== 软投票预测概率（前 5 个样本）===")
voting_soft.fit(X_train, y_train)
y_prob = voting_soft.predict_proba(X_test[:5])
y_pred = voting_soft.predict(X_test[:5])
for i in range(5):
    probs = ", ".join(f"{p:.3f}" for p in y_prob[i])
    print(f"  样本{i+1}: 预测={y_pred[i]}, 概率=[{probs}], 真实={y_test[i]}")

# ===================== 8. 投票法回归 =====================
print("\n=== Voting 回归 ===")
X_r, y_r = make_regression(n_samples=300, n_features=10, noise=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

reg_models = [
    ("dt", DecisionTreeRegressor(max_depth=5, random_state=42)),
    ("rf", RandomForestRegressor(n_estimators=50, random_state=42)),
    ("gb", GradientBoostingRegressor(n_estimators=50, random_state=42)),
    ("ridge", Ridge()),
]
voting_reg = VotingRegressor(estimators=reg_models)
voting_reg.fit(X_tr, y_tr)
print(f"Voting 回归 R²: {voting_reg.score(X_te, y_te):.4f}")
for name, model in reg_models:
    model.fit(X_tr, y_tr)
    print(f"  {name:>5} R²: {model.score(X_te, y_te):.4f}")

print("\n=== 投票法要点 ===")
print("- 硬投票: 多数类别获胜（不需要概率输出）")
print("- 软投票: 概率平均后选最大（需要模型支持 predict_proba）")
print("- 加权投票: 给更好的模型更高权重")
print("- 效果好当：模型多样性强且各模型准确率 > 随机")
print("- 简单有效，常作为 baseline 或最终集成手段")
