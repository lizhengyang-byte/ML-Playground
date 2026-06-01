"""
XGBoost —— 极端梯度提升，高效、正则化的 Boosting 实现，竞赛常胜算法
需要安装: pip install xgboost
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score, classification_report

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost 未安装，请运行: pip install xgboost\n")

# ===================== 1. 分类任务 =====================
if HAS_XGB:
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ===================== 2. 基础 XGBoost 分类 =====================
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("=== XGBoost 分类 ===")
    print(f"训练准确率: {xgb_clf.score(X_train, y_train):.4f}")
    print(f"测试准确率: {xgb_clf.score(X_test, y_test):.4f}")

    # ===================== 3. 特征重要性 =====================
    print("\n=== 特征重要性 ===")
    importances = xgb_clf.feature_importances_
    for i in np.argsort(importances)[::-1]:
        print(f"  特征{i}: {importances[i]:.4f}")

    # ===================== 4. 超参数调优 =====================
    print("\n=== GridSearchCV ===")
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [50, 100, 200],
    }
    gs = GridSearchCV(
        xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42),
        param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print(f"最佳参数: {gs.best_params_}")
    print(f"最佳 CV 准确率: {gs.best_score_:.4f}")
    print(f"测试准确率: {gs.best_estimator_.score(X_test, y_test):.4f}")

    # ===================== 5. early_stopping =====================
    print("\n=== Early Stopping ===")
    xgb_es = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        eval_metric="logloss", use_label_encoder=False, random_state=42,
    )
    xgb_es.fit(X_train, y_train, eval_set=[(X_test, y_test)],
               early_stopping_rounds=20, verbose=False)
    print(f"最优轮数: {xgb_es.best_iteration}")
    print(f"测试准确率: {xgb_es.score(X_test, y_test):.4f}")

    # ===================== 6. 回归任务 =====================
    print("\n=== XGBoost 回归 ===")
    X_r, y_r = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

    xgb_reg = xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
    )
    xgb_reg.fit(X_tr, y_tr)
    print(f"R²: {xgb_reg.score(X_te, y_te):.4f}")

# ===================== 7. XGBoost 核心特性 =====================
print("\n=== XGBoost 核心特性 ===")
print("1. 正则化: 目标函数包含 L1/L2 正则项，防止过拟合")
print("2. 二阶泰勒展开: 同时利用一阶梯度和二阶梯度（Hessian），优化更精确")
print("3. 列采样: 每棵树随机选择部分特征（类似随机森林）")
print("4. 缺失值处理: 自动学习缺失值的最优分裂方向")
print("5. 并行化: 特征级别并行（不是树级别）")
print("6. 缓存感知: 按列块组织数据，优化 CPU 缓存命中率")

print("\n=== XGBoost vs sklearn GradientBoosting ===")
print("- XGBoost 更快（C++ 实现 + 并行化）")
print("- XGBoost 有内置正则化")
print("- XGBoost 支持缺失值处理")
print("- XGBoost 有 early_stopping 和自定义评估指标")

print("\n=== XGBoost 要点 ===")
print("- max_depth: 通常 3~10，越大越容易过拟合")
print("- learning_rate: 通常 0.01~0.3，与 n_estimators 互补")
print("- subsample / colsample_bytree: 引入随机性防过拟合")
print("- min_child_weight: 控制叶节点最小样本权重")
