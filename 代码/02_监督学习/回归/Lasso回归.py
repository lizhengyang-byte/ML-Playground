"""
Lasso 回归 —— L1 正则化线性回归，可将部分系数压缩为 0，实现特征选择
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ===================== 1. 构造稀疏数据（只有少数特征有用）=====================
np.random.seed(42)
n, p = 200, 20
X = np.random.randn(n, p)
# 只有前 3 个特征真正有用
true_coef = np.zeros(p)
true_coef[0] = 5.0
true_coef[3] = -3.0
true_coef[7] = 2.0
y = X @ true_coef + np.random.randn(n) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== 2. 线性回归 vs Lasso =====================
lr = LinearRegression().fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

print("=== 线性回归 vs Lasso (alpha=0.1) ===")
print(f"{'':>20} {'线性回归':>12} {'Lasso':>12}")
print(f"{'训练 R²':>20} {lr.score(X_train, y_train):>12.4f} {lasso.score(X_train, y_train):>12.4f}")
print(f"{'测试 R²':>20} {lr.score(X_test, y_test):>12.4f} {lasso.score(X_test, y_test):>12.4f}")
print(f"非零系数: {np.count_nonzero(lr.coef_)} → {np.count_nonzero(lasso.coef_)}")

# ===================== 3. Lasso 的特征选择效果 =====================
print("\n=== Lasso 系数（对比真实值）===")
print(f"{'特征':>6} {'真实系数':>10} {'Lasso 系数':>12}")
for i in range(p):
    mark = " ← 真正有用的" if true_coef[i] != 0 else ""
    print(f"{'X'+str(i):>6} {true_coef[i]:>10.2f} {lasso.coef_[i]:>12.4f}{mark}")

# ===================== 4. alpha 参数影响 =====================
print("\n=== alpha 对稀疏性的影响 ===")
for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    l_a = Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train)
    n_nonzero = np.count_nonzero(l_a.coef_)
    print(f"  alpha={alpha:>5}: 非零系数={n_nonzero:>2}/{p}, "
          f"测试R²={l_a.score(X_test, y_test):.4f}")

# ===================== 5. LassoCV 自动选择 alpha =====================
print("\n=== LassoCV 交叉验证选 alpha ===")
lasso_cv = LassoCV(alphas=None, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train, y_train)
print(f"最佳 alpha: {lasso_cv.alpha_:.6f}")
print(f"测试 R²: {lasso_cv.score(X_test, y_test):.4f}")
print(f"非零系数: {np.count_nonzero(lasso_cv.coef_)}/{p}")

# ===================== 6. L1 vs L2 对比 =====================
from sklearn.linear_model import Ridge
print("\n=== L1 (Lasso) vs L2 (Ridge) 对比 ===")
ridge = Ridge(alpha=0.1).fit(X_train, y_train)
print(f"{'':>12} {'Ridge':>12} {'Lasso':>12}")
print(f"{'测试 R²':>12} {ridge.score(X_test, y_test):>12.4f} {lasso.score(X_test, y_test):>12.4f}")
print(f"{'非零系数':>12} {np.count_nonzero(ridge.coef_):>12} {np.count_nonzero(lasso.coef_):>12}")
print(f"{'系数L1范数':>12} {np.sum(np.abs(ridge.coef_)):>12.4f} {np.sum(np.abs(lasso.coef_)):>12.4f}")

print("\n=== Lasso 要点 ===")
print("- L1 正则化可将系数精确压缩为 0，实现自动特征选择")
print("- 适用于：高维数据、特征稀疏（只有少数特征真正相关）")
print("- alpha 越大，被压缩为 0 的系数越多")
print("- 注意：Lasso 最多选择 min(n, p) 个特征")
print("- 如果多个特征高度相关，Lasso 可能只选其中一个")
