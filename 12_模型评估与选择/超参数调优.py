"""
超参数调优 (Hyperparameter Tuning)

系统性地搜索最优超参数组合：
1. GridSearchCV - 网格搜索，穷举所有参数组合
2. RandomizedSearchCV - 随机搜索，从分布中采样参数
3. HalvingGridSearchCV - 逐步淘汰式网格搜索（资源高效）
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 生成数据
X, y = make_classification(
    n_samples=500, n_features=15, n_informative=8,
    n_classes=2, random_state=42
)

# 5折分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# 1. GridSearchCV 网格搜索
# ============================================================
print("=" * 60)
print("1. GridSearchCV 网格搜索")
print("=" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
}

# 计算总组合数
total_combos = 1
for v in param_grid.values():
    total_combos *= len(v)
print(f"参数组合总数: {total_combos}")
print(f"参数空间: {param_grid}")
print()

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
    return_train_score=True,
)

grid_search.fit(X, y)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
print(f"评估的参数组合数: {len(grid_search.cv_results_['params'])}")

# 显示Top5结果
print("\nTop5参数组合:")
results = grid_search.cv_results_
sorted_idx = np.argsort(results['rank_test_score'])[:5]
for idx in sorted_idx:
    print(f"  排名{results['rank_test_score'][idx]}: "
          f"验证分数={results['mean_test_score'][idx]:.4f} "
          f"训练分数={results['mean_train_score'][idx]:.4f} "
          f"参数={results['params'][idx]}")

# ============================================================
# 2. RandomizedSearchCV 随机搜索
# ============================================================
from scipy.stats import randint, uniform

print("\n" + "=" * 60)
print("2. RandomizedSearchCV 随机搜索")
print("=" * 60)

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9),
}

n_iter = 30  # 随机采样30次
print(f"采样次数: {n_iter}")
print(f"参数分布:")
for name, dist in param_distributions.items():
    print(f"  {name}: {type(dist).__name__}")
print()

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=0,
    return_train_score=True,
)

random_search.fit(X, y)

print(f"最佳参数: {random_search.best_params_}")
print(f"最佳交叉验证分数: {random_search.best_score_:.4f}")

print("\nTop5参数组合:")
results_r = random_search.cv_results_
sorted_idx_r = np.argsort(results_r['rank_test_score'])[:5]
for idx in sorted_idx_r:
    params_str = {k: v for k, v in results_r['params'][idx].items()
                  if isinstance(v, (int, float, str, type(None)))}
    print(f"  排名{results_r['rank_test_score'][idx]}: "
          f"验证分数={results_r['mean_test_score'][idx]:.4f} "
          f"参数={params_str}")

# ============================================================
# 3. 网格搜索 vs 随机搜索对比
# ============================================================
print("\n" + "=" * 60)
print("3. 网格搜索 vs 随机搜索对比")
print("=" * 60)

print(f"{'方法':<20} {'最佳分数':>10} {'最佳参数数':>10}")
print("-" * 45)
print(f"  {'网格搜索':<18} {grid_search.best_score_:>10.4f} {total_combos:>10}")
print(f"  {'随机搜索(30次)':<18} {random_search.best_score_:>10.4f} {n_iter:>10}")

# ============================================================
# 4. 不同模型的超参数调优
# ============================================================
print("\n" + "=" * 60)
print("4. 不同学习率的GBDT调优")
print("=" * 60)

from sklearn.ensemble import GradientBoostingClassifier

gbdt_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5],
}

gbdt_grid = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=gbdt_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
)

gbdt_grid.fit(X, y)

print(f"GBDT最佳参数: {gbdt_grid.best_params_}")
print(f"GBDT最佳分数: {gbdt_grid.best_score_:.4f}")

print("\n所有组合结果:")
results_gb = gbdt_grid.cv_results_
sorted_idx_gb = np.argsort(results_gb['rank_test_score'])
for rank, idx in enumerate(sorted_idx_gb):
    marker = " <-- 最佳" if rank == 0 else ""
    print(f"  排名{results_gb['rank_test_score'][idx]:>2}: "
          f"分数={results_gb['mean_test_score'][idx]:.4f} "
          f"参数={results_gb['params'][idx]}{marker}")
