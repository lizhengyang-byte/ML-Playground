"""
特征选择 (Feature Selection)

从原始特征中筛选出最有价值的子集，降低维度、减少过拟合、提升模型效率。

常用方法:
1. 方差阈值 (VarianceThreshold) - 去除低方差特征（近似常量）
2. 单变量统计检验 (SelectKBest) - 根据统计检验评分选择K个最佳特征
3. 互信息 (mutual_info_classif/regression) - 衡量特征与目标的非线性依赖
4. 递归特征消除 (RFE) - 反复训练模型并剔除最不重要特征
5. 基于模型的特征选择 (SelectFromModel) - 根据模型内置的feature_importances_筛选
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. 方差阈值 (VarianceThreshold)
# ============================================================
from sklearn.feature_selection import VarianceThreshold

print("=" * 60)
print("1. 方差阈值 (VarianceThreshold)")
print("=" * 60)

# 生成数据：10个特征，其中2个是近似常量（方差很小）
np.random.seed(42)
X_var = np.random.randn(200, 10)
X_var[:, 8] = 0.01  # 近似常量特征
X_var[:, 9] = 0.005  # 近似常量特征

print(f"原始特征数: {X_var.shape[1]}")
print(f"各特征方差: {np.round(X_var.var(axis=0), 4)}")

# 阈值设为0.1，方差低于此值的特征会被移除
selector_var = VarianceThreshold(threshold=0.1)
X_selected = selector_var.fit_transform(X_var)

print(f"方差阈值=0.1 后剩余特征数: {X_selected.shape[1]}")
print(f"被保留的特征索引: {selector_var.get_support(indices=True)}")
print(f"被移除的特征索引: {np.where(~selector_var.get_support())[0]}")

# ============================================================
# 2. 单变量统计检验 (SelectKBest + f_classif)
# ============================================================
from sklearn.feature_selection import SelectKBest, f_classif

print("\n" + "=" * 60)
print("2. 单变量统计检验 (SelectKBest + f_classif)")
print("=" * 60)

X_cls, y_cls = make_classification(
    n_samples=300, n_features=15, n_informative=5,
    n_redundant=3, n_classes=2, random_state=42
)

# 选择K=5个最佳特征
selector_kbest = SelectKBest(score_func=f_classif, k=5)
X_kbest = selector_kbest.fit_transform(X_cls, y_cls)

print(f"原始特征数: {X_cls.shape[1]}")
print(f"选择K=5后特征数: {X_kbest.shape[1]}")
print(f"各特征的F统计量: {np.round(selector_kbest.scores_, 2)}")
print(f"被选中的特征索引: {selector_kbest.get_support(indices=True)}")
print(f"被选中特征的p值: {np.round(selector_kbest.pvalues_[selector_kbest.get_support()], 6)}")

# ============================================================
# 3. 互信息 (mutual_info_classif)
# ============================================================
from sklearn.feature_selection import mutual_info_classif

print("\n" + "=" * 60)
print("3. 互信息 (mutual_info_classif)")
print("=" * 60)

# 互信息能捕捉非线性关系，适合更复杂的特征依赖
mi_scores = mutual_info_classif(X_cls, y_cls, random_state=42)

print("各特征的互信息得分:")
for i, score in enumerate(mi_scores):
    print(f"  特征{i}: {score:.4f}")

# 选择互信息得分最高的5个特征
top5_idx = np.argsort(mi_scores)[::-1][:5]
print(f"\n互信息Top5特征索引: {top5_idx}")
print(f"互信息Top5得分: {mi_scores[top5_idx].round(4)}")

# ============================================================
# 4. 递归特征消除 (RFE)
# ============================================================
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

print("\n" + "=" * 60)
print("4. 递归特征消除 (RFE)")
print("=" * 60)

# RFE反复训练模型，每次移除权重最小的特征
estimator = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=estimator, n_features_to_select=5, step=1)
X_rfe = rfe.fit_transform(X_cls, y_cls)

print(f"原始特征数: {X_cls.shape[1]}")
print(f"RFE选择后特征数: {X_rfe.shape[1]}")
print(f"各特征排名 (1=最佳): {rfe.ranking_}")
print(f"被选中的特征索引: {rfe.get_support(indices=True)}")
print(f"被移除的特征索引: {np.where(~rfe.get_support())[0]}")

# ============================================================
# 5. 基于模型的特征选择 (SelectFromModel)
# ============================================================
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

print("\n" + "=" * 60)
print("5. 基于模型的特征选择 (SelectFromModel)")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_cls, y_cls)

print(f"随机森林feature_importances_: {np.round(rf.feature_importances_, 4)}")

# 使用中位数作为阈值
selector_sfm = SelectFromModel(rf, threshold="median")
X_sfm = selector_sfm.fit_transform(X_cls, y_cls)

print(f"原始特征数: {X_cls.shape[1]}")
print(f"SelectFromModel选择后特征数: {X_sfm.shape[1]}")
print(f"被选中的特征索引: {selector_sfm.get_support(indices=True)}")

# 使用绝对值阈值
selector_sfm2 = SelectFromModel(rf, threshold=0.05)
X_sfm2 = selector_sfm2.fit_transform(X_cls, y_cls)
print(f"\n绝对阈值0.05选择后特征数: {X_sfm2.shape[1]}")
print(f"被选中的特征索引: {selector_sfm2.get_support(indices=True)}")

# ============================================================
# 综合对比
# ============================================================
print("\n" + "=" * 60)
print("综合对比：各方法选出的特征集")
print("=" * 60)
print(f"SelectKBest (F检验): {sorted(selector_kbest.get_support(indices=True))}")
print(f"互信息Top5:          {sorted(top5_idx)}")
print(f"RFE:                 {sorted(rfe.get_support(indices=True))}")
print(f"SelectFromModel(中位):{sorted(selector_sfm.get_support(indices=True))}")
print(f"SelectFromModel(0.05):{sorted(selector_sfm2.get_support(indices=True))}")
