"""
学习曲线 (Learning Curve)

通过改变训练集大小或模型复杂度，分析模型的学习状态：
1. learning_curve - 训练集大小 vs 模型性能
2. validation_curve - 超参数 vs 模型性能
3. 过拟合/欠拟合诊断
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 生成数据
X, y = make_classification(
    n_samples=500, n_features=15, n_informative=8,
    n_classes=2, random_state=42
)

# ============================================================
# 1. learning_curve 学习曲线
# ============================================================
print("=" * 60)
print("1. learning_curve - 训练集大小 vs 性能")
print("=" * 60)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy', random_state=42, n_jobs=-1
)

print(f"{'训练集大小':>10} {'训练准确率':>10} {'验证准确率':>10} {'差距':>8}")
print("-" * 45)
for size, t_mean, v_mean, t_std, v_std in zip(
    train_sizes,
    train_scores.mean(axis=1),
    val_scores.mean(axis=1),
    train_scores.std(axis=1),
    val_scores.std(axis=1)
):
    gap = t_mean - v_mean
    print(f"  {size:>8} {t_mean:>10.4f} {v_mean:>10.4f} {gap:>8.4f}")

# 诊断
final_train = train_scores.mean(axis=1)[-1]
final_val = val_scores.mean(axis=1)[-1]
gap_final = final_train - final_val

print(f"\n诊断结果:")
print(f"  最终训练准确率: {final_train:.4f}")
print(f"  最终验证准确率: {final_val:.4f}")
print(f"  训练-验证差距: {gap_final:.4f}")

if gap_final > 0.05 and final_val < 0.85:
    print("  -> 过拟合: 模型在训练集上表现好但泛化差，增加数据或简化模型")
elif final_val < 0.7 and final_train < 0.75:
    print("  -> 欠拟合: 模型容量不足，增加特征或使用更复杂的模型")
else:
    print("  -> 模型状态良好")

# ============================================================
# 2. 不同复杂度模型的学习曲线对比
# ============================================================
print("\n" + "=" * 60)
print("2. 不同复杂度模型的学习曲线对比")
print("=" * 60)

models_complexity = {
    '逻辑回归(简单)': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    '决策树(深度3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    '决策树(无限制)': DecisionTreeClassifier(max_depth=None, random_state=42),
    '随机森林(100棵)': RandomForestClassifier(n_estimators=100, random_state=42),
}

print(f"{'模型':<18} {'训练准确率':>10} {'验证准确率':>10} {'差距':>8} {'状态':>8}")
print("-" * 60)

for name, m in models_complexity.items():
    ts, tr_s, va_s = learning_curve(
        m, X, y,
        train_sizes=np.linspace(0.3, 1.0, 3),
        cv=5, scoring='accuracy', random_state=42
    )
    final_tr = tr_s.mean(axis=1)[-1]
    final_va = va_s.mean(axis=1)[-1]
    gap = final_tr - final_va

    if gap > 0.08:
        status = "过拟合"
    elif final_va < 0.7:
        status = "欠拟合"
    else:
        status = "良好"

    print(f"  {name:<16} {final_tr:>10.4f} {final_va:>10.4f} {gap:>8.4f} {status:>8}")

# ============================================================
# 3. validation_curve 验证曲线 (超参数搜索)
# ============================================================
print("\n" + "=" * 60)
print("3. validation_curve - 超参数 vs 性能")
print("=" * 60)

# 决策树最大深度
tree = DecisionTreeClassifier(random_state=42)
depths = [1, 2, 3, 5, 8, 10, 15, 20, None]

train_scores_depth, val_scores_depth = validation_curve(
    tree, X, y,
    param_name='max_depth',
    param_range=depths,
    cv=5, scoring='accuracy'
)

print("决策树 max_depth 验证曲线:")
print(f"{'max_depth':>10} {'训练准确率':>10} {'验证准确率':>10} {'差距':>8}")
print("-" * 45)
for depth, t_mean, v_mean in zip(depths, train_scores_depth.mean(axis=1), val_scores_depth.mean(axis=1)):
    depth_str = str(depth) if depth is not None else "None"
    gap = t_mean - v_mean
    print(f"  {depth_str:>8} {t_mean:>10.4f} {v_mean:>10.4f} {gap:>8.4f}")

# 找到最佳深度
best_idx = np.argmax(val_scores_depth.mean(axis=1))
best_depth = depths[best_idx]
print(f"\n最佳max_depth: {best_depth} (验证准确率: {val_scores_depth.mean(axis=1)[best_idx]:.4f})")

# ============================================================
# 4. 随机森林树数量验证曲线
# ============================================================
print("\n" + "=" * 60)
print("4. 随机森林 n_estimators 验证曲线")
print("=" * 60)

rf = RandomForestClassifier(random_state=42)
n_estimators_range = [5, 10, 25, 50, 100, 200]

train_scores_ne, val_scores_ne = validation_curve(
    rf, X, y,
    param_name='n_estimators',
    param_range=n_estimators_range,
    cv=5, scoring='accuracy'
)

print(f"{'n_estimators':>12} {'训练准确率':>10} {'验证准确率':>10} {'差距':>8}")
print("-" * 45)
for n_est, t_mean, v_mean in zip(
    n_estimators_range,
    train_scores_ne.mean(axis=1),
    val_scores_ne.mean(axis=1)
):
    gap = t_mean - v_mean
    print(f"  {n_est:>10} {t_mean:>10.4f} {v_mean:>10.4f} {gap:>8.4f}")

# ============================================================
# 5. 综合诊断
# ============================================================
print("\n" + "=" * 60)
print("5. 过拟合/欠拟合诊断总结")
print("=" * 60)

diagnostics = {
    '逻辑回归': {'train': 0.82, 'val': 0.80},
    '决策树(浅)': {'train': 0.85, 'val': 0.82},
    '决策树(深)': {'train': 1.00, 'val': 0.75},
    '随机森林': {'train': 0.95, 'val': 0.88},
}

print(f"{'模型':<14} {'训练':>6} {'验证':>6} {'差距':>6} {'诊断':<12}")
print("-" * 50)
for name, scores in diagnostics.items():
    gap = scores['train'] - scores['val']
    if gap > 0.15:
        diag = "严重过拟合"
    elif gap > 0.05:
        diag = "轻微过拟合"
    elif scores['val'] < 0.7:
        diag = "欠拟合"
    else:
        diag = "良好"
    print(f"  {name:<12} {scores['train']:>6.2f} {scores['val']:>6.2f} {gap:>6.2f} {diag}")

print("""
诊断规则:
  过拟合信号:
    - 训练准确率 >> 验证准确率 (差距大)
    - 训练准确率接近1.0，验证准确率明显低
    - 增加数据量后验证准确率仍在提升
  欠拟合信号:
    - 训练和验证准确率都很低
    - 增加数据量后性能不再提升
  解决方案:
    过拟合: 增加数据、正则化、简化模型、特征选择、Dropout
    欠拟合: 增加特征、使用更复杂模型、减少正则化、集成方法
""")
