"""
决策树分类 —— 基于特征划分的树形分类模型，可解释性强
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# ===================== 1. 加载数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===================== 2. 基础决策树 =====================
# criterion: "gini"(基尼系数) 或 "entropy"(信息增益)
dt = DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=42)
dt.fit(X_train, y_train)

print("=== 决策树分类（默认参数）===")
print(f"训练集准确率: {dt.score(X_train, y_train):.4f}")
print(f"测试集准确率: {dt.score(X_test, y_test):.4f}")
print(f"树深度: {dt.get_depth()}")
print(f"叶节点数: {dt.get_n_leaves()}")
print(f"特征重要性: {dict(zip(iris.feature_names, dt.feature_importances_.round(4)))}")

# ===================== 3. 文本形式展示树结构 =====================
print("\n=== 树结构（文本）===")
tree_rules = export_text(dt, feature_names=list(iris.feature_names), decimals=3)
print(tree_rules)

# ===================== 4. 剪枝控制复杂度 =====================
print("\n=== 不同 max_depth 对比 ===")
for depth in [1, 2, 3, 5, 10, None]:
    dt_d = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_d.fit(X_train, y_train)
    train_acc = dt_d.score(X_train, y_train)
    test_acc = dt_d.score(X_test, y_test)
    depth_str = str(depth) if depth else "无限制"
    print(f"max_depth={depth_str:>3}: 训练={train_acc:.4f}, 测试={test_acc:.4f}, "
          f"叶节点={dt_d.get_n_leaves()}")

# ===================== 5. 超参数搜索 =====================
print("\n=== GridSearchCV 超参数搜索 ===")
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 3, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
gs.fit(X_train, y_train)
print(f"最佳参数: {gs.best_params_}")
print(f"最佳交叉验证准确率: {gs.best_score_:.4f}")
print(f"测试集准确率: {gs.best_estimator_.score(X_test, y_test):.4f}")

# ===================== 6. 预测概率 =====================
print("\n=== 预测概率示例 ===")
dt_best = gs.best_estimator_
y_prob = dt_best.predict_proba(X_test[:5])
y_pred = dt_best.predict(X_test[:5])
for i in range(5):
    probs = ", ".join(f"{p:.3f}" for p in y_prob[i])
    print(f"样本{i+1}: 预测={iris.target_names[y_pred[i]]}, 概率=[{probs}], "
          f"真实={iris.target_names[y_test[i]]}")

y_pred_all = dt_best.predict(X_test)
print(f"\n分类报告:\n{classification_report(y_test, y_pred_all, target_names=iris.target_names)}")
