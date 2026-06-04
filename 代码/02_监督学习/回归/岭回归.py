"""
岭回归（Ridge Regression）—— 在线性回归基础上加入 L2 正则化，缓解多重共线性
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ===================== 1. 生成含共线性的数据 =====================
np.random.seed(42)
n, p = 200, 10
X = np.random.randn(n, p)
# 让部分特征高度相关（模拟多重共线性）
X[:, 1] = X[:, 0] * 0.9 + np.random.randn(n) * 0.1
X[:, 2] = X[:, 0] * 0.8 + np.random.randn(n) * 0.1
y = 3 * X[:, 0] + 2 * X[:, 3] - 1.5 * X[:, 5] + np.random.randn(n) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== 2. 线性回归 vs 岭回归 =====================
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)

print("=== 线性回归 vs 岭回归 ===")
print(f"{'':>20} {'线性回归':>12} {'岭回归':>12}")
print(f"{'训练 R²':>20} {lr.score(X_train, y_train):>12.4f} {ridge.score(X_train, y_train):>12.4f}")
print(f"{'测试 R²':>20} {lr.score(X_test, y_test):>12.4f} {ridge.score(X_test, y_test):>12.4f}")

print(f"\n系数对比:")
print(f"{'特征':>8} {'线性回归':>12} {'岭回归':>12}")
for i in range(p):
    print(f"{'X'+str(i):>8} {lr.coef_[i]:>12.4f} {ridge.coef_[i]:>12.4f}")

# ===================== 3. alpha 参数选择 =====================
print("\n=== alpha 参数影响 ===")
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
for alpha in alphas:
    r = Ridge(alpha=alpha).fit(X_train, y_train)
    print(f"  alpha={alpha:>8}: 训练R²={r.score(X_train, y_train):.4f}, "
          f"测试R²={r.score(X_test, y_test):.4f}, 系数L2范数={np.linalg.norm(r.coef_):.4f}")

# ===================== 4. 交叉验证选 alpha =====================
# RidgeCV 自动在多个 alpha 上做交叉验证
ridge_cv = RidgeCV(alphas=alphas, scoring="r2", cv=5)
ridge_cv.fit(X_train, y_train)
print(f"\n=== 交叉验证最优 alpha ===")
print(f"最佳 alpha: {ridge_cv.alpha_}")
print(f"测试 R²: {ridge_cv.score(X_test, y_test):.4f}")

# ===================== 5. 正则化原理 =====================
print("\n=== 岭回归原理 ===")
print("损失函数: ||y - Xw||² + α||w||²")
print("α=0 退化为普通线性回归；α→∞ 所有系数趋于 0")
print("L2 正则化将系数收缩向 0，但不会精确等于 0（不做特征选择）")

print("\n=== 岭回归要点 ===")
print("- 适用于：多重共线性、特征数 > 样本数、系数过大")
print("- alpha 越大正则化越强，模型越简单（偏差大、方差小）")
print("- 不能做特征选择（系数不会精确为 0）")
print("- 特征缩放很重要（正则化对系数大小敏感）")
