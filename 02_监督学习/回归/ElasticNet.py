"""
ElasticNet 回归 —— 结合 L1 (Lasso) 和 L2 (Ridge) 正则化的线性回归
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ===================== 1. 构造数据 =====================
np.random.seed(42)
n, p = 200, 15
X = np.random.randn(n, p)
true_coef = np.zeros(p)
true_coef[:3] = [5, -3, 2]  # 只有前 3 个有用
true_coef[3] = 1.5
y = X @ true_coef + np.random.randn(n) * 0.5

# 添加高度相关的特征组
X[:, 10] = X[:, 0] * 0.9 + np.random.randn(n) * 0.05
X[:, 11] = X[:, 1] * 0.85 + np.random.randn(n) * 0.05

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===================== 2. 三种回归对比 =====================
lr = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)
en = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X_train, y_train)

print("=== 四种回归对比 ===")
print(f"{'方法':>18} {'测试 R²':>10} {'非零系数':>10}")
print(f"{'线性回归':>18} {lr.score(X_test, y_test):>10.4f} {np.count_nonzero(lr.coef_):>10}")
print(f"{'Ridge (L2)':>18} {ridge.score(X_test, y_test):>10.4f} {np.count_nonzero(ridge.coef_):>10}")
print(f"{'Lasso (L1)':>18} {lasso.score(X_test, y_test):>10.4f} {np.count_nonzero(lasso.coef_):>10}")
print(f"{'ElasticNet':>18} {en.score(X_test, y_test):>10.4f} {np.count_nonzero(en.coef_):>10}")

# ===================== 3. l1_ratio 参数 =====================
# l1_ratio=1 等价于 Lasso, l1_ratio=0 等价于 Ridge
print("\n=== l1_ratio 对比（alpha=0.1）===")
for l1_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    en_r = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10000)
    en_r.fit(X_train, y_train)
    n_nz = np.count_nonzero(en_r.coef_)
    name = "Ridge" if l1_ratio == 0 else ("Lasso" if l1_ratio == 1 else f"EN")
    print(f"  l1_ratio={l1_ratio}: 测试R²={en_r.score(X_test, y_test):.4f}, "
          f"非零系数={n_nz}/{p}")

# ===================== 4. 超参数网格搜索 =====================
print("\n=== GridSearchCV (alpha + l1_ratio) ===")
param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}
gs = GridSearchCV(
    ElasticNet(max_iter=10000, random_state=42),
    param_grid, cv=5, scoring="r2", n_jobs=-1
)
gs.fit(X_train, y_train)
print(f"最佳参数: {gs.best_params_}")
print(f"最佳交叉验证 R²: {gs.best_score_:.4f}")
print(f"测试集 R²: {gs.best_estimator_.score(X_test, y_test):.4f}")

# ===================== 5. ElasticNetCV =====================
print("\n=== ElasticNetCV ===")
en_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, random_state=42, max_iter=10000)
en_cv.fit(X_train, y_train)
print(f"最佳 alpha: {en_cv.alpha_:.6f}")
print(f"最佳 l1_ratio: {en_cv.l1_ratio_}")
print(f"测试 R²: {en_cv.score(X_test, y_test):.4f}")

print("\n=== ElasticNet 要点 ===")
print("损失函数: ||y - Xw||² + α × [l1_ratio × ||w||₁ + 0.5 × (1-l1_ratio) × ||w||²]")
print("- l1_ratio=1 → 纯 Lasso (L1)，可做特征选择")
print("- l1_ratio=0 → 纯 Ridge (L2)，处理共线性")
print("- 0<l1_ratio<1 → 两者兼顾，尤其适合特征高度相关时")
