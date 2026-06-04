"""
LightGBM —— 微软提出的高效梯度提升框架,基于直方图的决策树算法
需要安装: pip install lightgbm
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[SKIP] LightGBM 未安装，跳过本示例")
    import sys; sys.exit(0)
    HAS_LGB = False
    print("LightGBM 未安装,请运行: pip install lightgbm\n")

if HAS_LGB:
    # ===================== 1. 分类任务 =====================
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                               n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ===================== 2. 基础 LightGBM =====================
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    lgb_clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    print("=== LightGBM 分类 ===")
    print(f"训练准确率: {lgb_clf.score(X_train, y_train):.4f}")
    print(f"测试准确率: {lgb_clf.score(X_test, y_test):.4f}")
    print(f"最优轮数: {lgb_clf.best_iteration_}")

    # ===================== 3. 特征重要性 =====================
    print("\n=== 特征重要性 (gain) ===")
    importances = lgb_clf.feature_importances_
    for i in np.argsort(importances)[::-1]:
        print(f"  特征{i}: {importances[i]:.4f}")

    # ===================== 4. 关键参数 =====================
    print("\n=== 关键参数 ===")
    print("num_leaves: 叶子数（核心参数）,比 max_depth 控制更细")
    print("max_depth: 树的最大深度（-1=不限）")
    print("min_data_in_leaf: 叶节点最小样本数,防过拟合")

    # ===================== 5. num_leaves vs max_depth =====================
    print("\n=== num_leaves 对比 ===")
    for nl in [7, 15, 31, 63, 127]:
        m = lgb.LGBMClassifier(n_estimators=100, num_leaves=nl, random_state=42, verbose=-1)
        m.fit(X_train, y_train)
        print(f"  num_leaves={nl:>3}: 测试准确率={m.score(X_test, y_test):.4f}")

    # ===================== 6. 回归任务 =====================
    print("\n=== LightGBM 回归 ===")
    X_r, y_r = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    lgb_reg = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    lgb_reg.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    print(f"R²: {lgb_reg.score(X_te, y_te):.4f}")

    # ===================== 7. LightGBM vs XGBoost =====================
    print("\n=== LightGBM vs XGBoost ===")
    print("Leaf-wise 生长: LightGBM 每次选增益最大的叶子分裂（非层级生长）")
    print("直方图加速: 将连续值分桶,减少分裂点搜索时间")
    print("单边梯度采样 (GOSS): 保留大梯度样本,随机采样小梯度样本")
    print("互斥特征捆绑 (EFB): 将互斥稀疏特征合并")

print("\n=== LightGBM 要点 ===")
print("- num_leaves 是最核心的参数（控制模型复杂度）")
print("- 比 XGBoost 快 2~10 倍（大数据集优势明显）")
print("- 内存占用更低（直方图算法）")
print("- 更适合大规模数据和高维特征")
print("- 缺点：小数据集可能不如 XGBoost 稳定")
