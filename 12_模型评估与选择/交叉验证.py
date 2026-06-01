"""
交叉验证 (Cross-Validation)

将数据分为训练集和验证集，多次评估模型以获得更稳定的性能估计：
1. KFold - 基础K折交叉验证
2. StratifiedKFold - 分层K折（保持类别比例）
3. cross_val_score - 快速获取交叉验证分数
4. cross_validate - 返回多个评估指标和时间信息
5. learning_curve - 学习曲线分析数据量对模型性能的影响
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, cross_validate, learning_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 生成数据
X, y = make_classification(
    n_samples=300, n_features=10, n_informative=5,
    n_classes=3, n_clusters_per_class=1, random_state=42
)

model = LogisticRegression(max_iter=1000, random_state=42)

# ============================================================
# 1. KFold 基础K折交叉验证
# ============================================================
print("=" * 60)
print("1. KFold 基础K折交叉验证")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"数据总量: {len(X)}, K={kf.n_splits}")
print()

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"第{fold_idx+1}折: 训练集={len(train_idx)}样本, 验证集={len(val_idx)}样本")
    print(f"  训练集标签分布: {np.bincount(y[train_idx])}")
    print(f"  验证集标签分布: {np.bincount(y[val_idx])}")

# ============================================================
# 2. StratifiedKFold 分层K折
# ============================================================
print("\n" + "=" * 60)
print("2. StratifiedKFold 分层K折")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("分层K折保持每折的类别比例与整体一致:")
print(f"整体类别比例: {np.bincount(y) / len(y)}")
print()

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"第{fold_idx+1}折: 训练集={len(train_idx)}, 验证集={len(val_idx)}")
    print(f"  训练集类别比例: {np.round(np.bincount(y[train_idx]) / len(train_idx), 3)}")
    print(f"  验证集类别比例: {np.round(np.bincount(y[val_idx]) / len(val_idx), 3)}")

# ============================================================
# 3. cross_val_score 快速交叉验证
# ============================================================
print("\n" + "=" * 60)
print("3. cross_val_score 快速交叉验证")
print("=" * 60)

# 单指标交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"5折交叉验证准确率: {scores}")
print(f"平均准确率: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 多种指标
for metric in ['accuracy', 'f1_weighted', 'precision_weighted']:
    scores_m = cross_val_score(model, X, y, cv=5, scoring=metric)
    print(f"{metric}: {scores_m.mean():.4f} (+/- {scores_m.std():.4f})")

# ============================================================
# 4. cross_validate 多指标评估
# ============================================================
print("\n" + "=" * 60)
print("4. cross_validate 多指标 + 训练/测试时间")
print("=" * 60)

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1_weighted',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
}

cv_results = cross_validate(
    model, X, y, cv=5, scoring=scoring,
    return_train_score=True, return_estimator=True
)

print(f"{'指标':<25} {'训练集均值':>10} {'验证集均值':>10} {'验证集标准差':>12}")
print("-" * 60)
for key in scoring:
    train_key = f'train_{key}'
    test_key = f'test_{key}'
    if train_key in cv_results and test_key in cv_results:
        train_mean = cv_results[train_key].mean()
        test_mean = cv_results[test_key].mean()
        test_std = cv_results[test_key].std()
        print(f"  {key:<23} {train_mean:>10.4f} {test_mean:>10.4f} {test_std:>12.4f}")

print(f"\n训练时间 (5折总计): {cv_results['fit_time'].sum():.3f}秒")
print(f"预测时间 (5折总计): {cv_results['score_time'].sum():.3f}秒")

# ============================================================
# 5. 不同模型的交叉验证对比
# ============================================================
print("\n" + "=" * 60)
print("5. 不同模型的交叉验证对比")
print("=" * 60)

models = {
    '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
    '随机森林(10棵树)': RandomForestClassifier(n_estimators=10, random_state=42),
    '随机森林(100棵树)': RandomForestClassifier(n_estimators=100, random_state=42),
}

print(f"{'模型':<20} {'准确率均值':>10} {'准确率标准差':>12} {'F1均值':>10}")
print("-" * 55)
for name, m in models.items():
    acc_scores = cross_val_score(m, X, y, cv=5, scoring='accuracy')
    f1_scores = cross_val_score(m, X, y, cv=5, scoring='f1_weighted')
    print(f"  {name:<18} {acc_scores.mean():>10.4f} {acc_scores.std():>12.4f} {f1_scores.mean():>10.4f}")

# ============================================================
# 6. 学习曲线
# ============================================================
print("\n" + "=" * 60)
print("6. learning_curve 学习曲线")
print("=" * 60)

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='accuracy', random_state=42
)

print(f"{'训练集大小':>10} {'训练集准确率':>12} {'验证集准确率':>12}")
print("-" * 38)
for size, train_mean, val_mean in zip(
    train_sizes,
    train_scores.mean(axis=1),
    val_scores.mean(axis=1)
):
    print(f"  {size:>8} {train_mean:>12.4f} {val_mean:>12.4f}")

print("\n分析:")
gap = train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]
print(f"  最终训练-验证差距: {gap:.4f}")
if gap > 0.05:
    print("  -> 存在过拟合迹象 (训练集远高于验证集)")
elif val_scores.mean(axis=1)[-1] < 0.6:
    print("  -> 可能欠拟合 (整体准确率偏低)")
else:
    print("  -> 模型表现良好")
