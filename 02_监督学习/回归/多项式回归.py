"""
多项式回归 —— 通过多项式特征扩展捕捉非线性关系
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# ===================== 1. 生成非线性数据 =====================
np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)
y = 0.5 * X.ravel() ** 3 - 2 * X.ravel() ** 2 + 3 * X.ravel() + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== 2. 不同阶数的多项式回归 =====================
print("=== 不同阶数的多项式回归 ===")
for degree in [1, 2, 3, 4, 5, 10]:
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr", LinearRegression())
    ])
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    n_features = model.named_steps["poly"].n_output_features_
    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"  degree={degree:>2}: 特征数={n_features:>3}, 训练R²={train_r2:.4f}, "
          f"测试R²={test_r2:.4f}, CV-R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

# ===================== 3. 特征扩展效果 =====================
print("\n=== PolynomialFeatures 特征扩展 ===")
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(X[:5])
print(f"原始特征 (degree=1): {X[:5].ravel()}")
print(f"扩展后 (degree=2): {poly2.get_feature_names_out(['x'])}")
print(f"值:\n{X_poly2[:3].round(4)}")

poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly3.fit_transform(X[:3])
print(f"\n扩展后 (degree=3): {poly3.get_feature_names_out(['x'])}")

# ===================== 4. Pipeline 构建完整流程 =====================
print("\n=== Pipeline: 缩放 + 多项式 + 回归 ===")
pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("lr", LinearRegression())
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(f"degree=3 测试 MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"degree=3 测试 R²: {r2_score(y_test, y_pred):.4f}")

# ===================== 5. 前 10 个预测对比 =====================
print("\n=== 预测值 vs 真实值（前 10 个）===")
for i in range(10):
    print(f"  真实={y_test[i]:>8.2f}, 预测={y_pred[i]:>8.2f}")

print("\n=== 多项式回归要点 ===")
print("- 通过 PolynomialFeatures 扩展特征空间，将非线性问题转化为线性问题")
print("- degree 越高模型越复杂，容易过拟合（高阶项噪声放大）")
print("- 一般 degree=2 或 3 足够，degree>5 很少使用")
print("- 多项式回归的参数数量随维度指数增长（维度灾难）")
print("- 推荐用 Pipeline 封装，避免数据泄露")
