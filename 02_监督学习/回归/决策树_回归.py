"""
决策树回归 —— 基于特征划分的树形回归模型，输出叶节点的均值
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, r2_score

# ===================== 1. 生成数据 =====================
X, y = make_regression(n_samples=300, n_features=5, n_informative=3, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== 2. 基础决策树回归 =====================
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

print("=== 决策树回归 ===")
print(f"训练 R²: {dt.score(X_train, y_train):.4f}")
print(f"测试 R²: {dt.score(X_test, y_test):.4f}")
print(f"树深度: {dt.get_depth()}")
print(f"叶节点数: {dt.get_n_leaves()}")
print(f"特征重要性: {dt.feature_importances_.round(4)}")

# ===================== 3. 文本树结构 =====================
print("\n=== 树结构（前 20 行）===")
tree_rules = export_text(dt, feature_names=[f"X{i}" for i in range(5)], decimals=2)
print("\n".join(tree_rules.split("\n")[:20]))

# ===================== 4. max_depth 过拟合控制 =====================
print("\n=== max_depth 影响 ===")
for depth in [2, 3, 5, 8, 10, 15, None]:
    dt_d = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt_d.fit(X_train, y_train)
    train_r2 = dt_d.score(X_train, y_train)
    test_r2 = dt_d.score(X_test, y_test)
    depth_str = str(depth) if depth else "无限制"
    print(f"  depth={depth_str:>3}: 训练R²={train_r2:.4f}, 测试R²={test_r2:.4f}, "
          f"叶节点={dt_d.get_n_leaves()}")

# ===================== 5. 超参数搜索 =====================
print("\n=== GridSearchCV ===")
param_grid = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
gs = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid, cv=5, scoring="r2", n_jobs=-1
)
gs.fit(X_train, y_train)
print(f"最佳参数: {gs.best_params_}")
print(f"最佳 CV R²: {gs.best_score_:.4f}")
print(f"测试 R²: {gs.best_estimator_.score(X_test, y_test):.4f}")

# ===================== 6. 预测示例 =====================
y_pred = gs.best_estimator_.predict(X_test)
print(f"\n=== 预测值 vs 真实值（前 10 个）===")
print(f"测试 MSE: {mean_squared_error(y_test, y_pred):.4f}")
for i in range(10):
    print(f"  真实={y_test[i]:>8.2f}, 预测={y_pred[i]:>8.2f}, 残差={y_test[i]-y_pred[i]:>8.2f}")

print("\n=== 决策树回归要点 ===")
print("- 叶节点输出该节点所有训练样本的均值")
print("- 天然可处理非线性关系，不需要特征缩放")
print("- 极易过拟合（无限制深度时训练 R²=1.0）")
print("- 需要剪枝（限制 max_depth / min_samples_leaf）")
print("- 方差大：数据微小变化可能导致完全不同的树")
