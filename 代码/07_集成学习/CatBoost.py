"""
CatBoost —— Yandex 提出的梯度提升框架,原生支持类别特征,抗过拟合能力强
需要安装: pip install catboost
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("[SKIP] CatBoost 未安装，跳过本示例")
    import sys; sys.exit(0)
    HAS_CB = False
    print("CatBoost 未安装,请运行: pip install catboost\n")

if HAS_CB:
    # ===================== 1. 分类任务 =====================
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ===================== 2. 基础 CatBoost =====================
    cb_clf = CatBoostClassifier(
        iterations=100,
        depth=3,
        learning_rate=0.1,
        random_seed=42,
        verbose=0,
    )
    cb_clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

    print("=== CatBoost 分类 ===")
    print(f"训练准确率: {cb_clf.score(X_train, y_train):.4f}")
    print(f"测试准确率: {cb_clf.score(X_test, y_test):.4f}")
    print(f"最优迭代次数: {cb_clf.best_iteration_}")

    # ===================== 3. 特征重要性 =====================
    print("\n=== 特征重要性 ===")
    importances = cb_clf.feature_importances_
    for i in np.argsort(importances)[::-1]:
        print(f"  特征{i}: {importances[i]:.4f}")

    # ===================== 4. 类别特征支持 =====================
    print("\n=== 类别特征处理 ===")
    # 构造含类别特征的数据
    np.random.seed(42)
    df = pd.DataFrame({
        "数值1": np.random.randn(200),
        "数值2": np.random.randn(200),
        "城市": np.random.choice(["北京", "上海", "广州"], 200),
        "性别": np.random.choice(["男", "女"], 200),
    })
    y_cat = (df["数值1"] + df["数值2"] > 0).astype(int)

    cat_features = ["城市", "性别"]
    cb_cat = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=42)
    cb_cat.fit(df, y_cat, cat_features=cat_features)
    print(f"含类别特征的准确率: {cb_cat.score(df, y_cat):.4f}")

    # ===================== 5. 有序目标编码 =====================
    print("\n=== CatBoost 的 Ordered Boosting ===")
    print("CatBoost 的核心创新:")
    print("1. Ordered Boosting: 训练第 t 棵树时只用时间上在它之前的数据预测残差")
    print("   避免 target leakage（目标泄露）")
    print("2. Ordered TS: 类别特征的目标编码也是按时间顺序计算的")
    print("   而非用全部数据的全局均值（避免过拟合）")

    # ===================== 6. 回归任务 =====================
    print("\n=== CatBoost 回归 ===")
    X_r, y_r = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    cb_reg = CatBoostRegressor(iterations=100, depth=3, verbose=0, random_seed=42)
    cb_reg.fit(X_tr, y_tr, eval_set=(X_te, y_te), early_stopping_rounds=50)
    print(f"R²: {cb_reg.score(X_te, y_te):.4f}")

    # ===================== 7. 不同 depth 对比 =====================
    print("\n=== depth 对比 ===")
    for d in [2, 3, 4, 5, 6, 8]:
        m = CatBoostClassifier(iterations=100, depth=d, verbose=0, random_seed=42)
        m.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
        print(f"  depth={d}: 测试准确率={m.score(X_test, y_test):.4f}")

print("\n=== CatBoost 要点 ===")
print("- 原生类别特征支持（无需手动编码）")
print("- Ordered Boosting: 防止 target leakage,抗过拟合能力强")
print("- 默认参数通常就能取得很好的效果（调参需求低）")
print("- 对缺失值和类别特征的处理是内置的")
print("- 训练速度介于 XGBoost 和 LightGBM 之间")
print("- 适合：含大量类别特征的结构化数据（如推荐、广告）")
