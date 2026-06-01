"""
ROC曲线 (ROC Curve)

ROC曲线和AUC用于评估分类器在不同阈值下的性能：
1. 二分类ROC曲线 - TPR vs FPR
2. 多分类ROC曲线 - One-vs-Rest策略
3. Precision-Recall曲线 - 类别不平衡时更合适
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# ============================================================
# 1. 二分类ROC曲线
# ============================================================
print("=" * 60)
print("1. 二分类ROC曲线")
print("=" * 60)

X_binary, y_binary = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_classes=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# 两个模型对比
models = {
    '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    print(f"\n{name}:")
    print(f"  AUC = {roc_auc:.4f}")

    # 打印关键阈值点
    print(f"  {'阈值':>8} {'FPR':>8} {'TPR':>8} {'1-特异度':>8} {'灵敏度':>8}")
    # 选择5个代表性阈值点
    n_points = len(thresholds)
    indices = np.linspace(0, n_points - 1, 5, dtype=int)
    for idx in indices:
        print(f"  {thresholds[idx]:>8.4f} {fpr[idx]:>8.4f} {tpr[idx]:>8.4f} {fpr[idx]:>8.4f} {tpr[idx]:>8.4f}")

    # 找到最优阈值 (Youden's J statistic)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    print(f"  最优阈值 (Youden J): {best_threshold:.4f}")
    print(f"  对应TPR: {tpr[best_idx]:.4f}, FPR: {fpr[best_idx]:.4f}")

# ============================================================
# 2. 多分类ROC曲线 (One-vs-Rest)
# ============================================================
print("\n" + "=" * 60)
print("2. 多分类ROC曲线 (One-vs-Rest)")
print("=" * 60)

X_multi, y_multi = make_classification(
    n_samples=600, n_features=10, n_informative=5,
    n_classes=4, n_clusters_per_class=1, random_state=42
)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42, stratify=y_multi
)

model_multi = LogisticRegression(max_iter=1000, random_state=42)
model_multi.fit(X_train_m, y_train_m)
y_proba_m = model_multi.predict_proba(X_test_m)

n_classes = len(np.unique(y_multi))
class_names = [f"类{i}" for i in range(n_classes)]

print(f"类别数: {n_classes}")
print()

# One-vs-Rest: 每个类别 vs 其他所有类别
print("One-vs-Rest ROC曲线:")
for i in range(n_classes):
    fpr_i, tpr_i, _ = roc_curve(y_test_m == i, y_proba_m[:, i])
    auc_i = auc(fpr_i, tpr_i)
    print(f"  {class_names[i]} vs 其他: AUC = {auc_i:.4f}")

# 整体多分类AUC
auc_ovr = roc_auc_score(y_test_m, y_proba_m, multi_class='ovr', average='macro')
auc_ovo = roc_auc_score(y_test_m, y_proba_m, multi_class='ovo', average='macro')
print(f"\n  宏平均AUC (One-vs-Rest): {auc_ovr:.4f}")
print(f"  宏平均AUC (One-vs-One):  {auc_ovo:.4f}")

# ============================================================
# 3. Precision-Recall曲线
# ============================================================
print("\n" + "=" * 60)
print("3. Precision-Recall曲线")
print("=" * 60)

# 使用二分类数据
model_pr = LogisticRegression(max_iter=1000, random_state=42)
model_pr.fit(X_train, y_train)
y_proba_pr = model_pr.predict_proba(X_test)[:, 1]

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba_pr)
ap = average_precision_score(y_test, y_proba_pr)

print(f"平均精度 (Average Precision): {ap:.4f}")
print(f"\nPR曲线关键点:")
print(f"  {'阈值':>8} {'精确率':>8} {'召回率':>8}")

# 选择代表性点
n_pr = len(thresholds_pr)
indices_pr = np.linspace(0, n_pr - 1, 6, dtype=int)
for idx in indices_pr:
    if idx < len(thresholds_pr):
        print(f"  {thresholds_pr[idx]:>8.4f} {precision[idx]:>8.4f} {recall[idx]:>8.4f}")

# 在不同召回率水平下的精确率
print(f"\n不同召回率水平下的精确率:")
for target_recall in [0.5, 0.6, 0.7, 0.8, 0.9]:
    idx = np.searchsorted(-recall, -target_recall)
    if idx < len(precision):
        print(f"  召回率>={target_recall:.1f}时, 最高精确率: {precision[idx]:.4f}")

# ============================================================
# 4. ROC vs PR 曲线对比
# ============================================================
print("\n" + "=" * 60)
print("4. ROC曲线 vs PR曲线")
print("=" * 60)

# 不同正负比例下的对比
print(f"\n原始数据正负比例: {np.bincount(y_test)}")
print(f"  正类比例: {y_test.mean():.4f}")

# 模拟不平衡数据
X_imb, y_imb = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_classes=2, weights=[0.95, 0.05], random_state=42
)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb
)

model_imb = LogisticRegression(max_iter=1000, random_state=42)
model_imb.fit(X_train_imb, y_train_imb)
y_proba_imb = model_imb.predict_proba(X_test_imb)[:, 1]

fpr_imb, tpr_imb, _ = roc_curve(y_test_imb, y_proba_imb)
precision_imb, recall_imb, _ = precision_recall_curve(y_test_imb, y_proba_imb)

auc_roc_imb = auc(fpr_imb, tpr_imb)
ap_imb = average_precision_score(y_test_imb, y_proba_imb)

print(f"\n不平衡数据 (正类仅5%):")
print(f"  AUC-ROC: {auc_roc_imb:.4f}")
print(f"  Average Precision: {ap_imb:.4f}")
print(f"  说明: 不平衡时ROC可能过于乐观，PR曲线更真实反映模型对少数类的识别能力")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
ROC曲线:
  - 横轴: FPR = FP / (FP + TN)，假阳性率
  - 纵轴: TPR = TP / (TP + FN)，真阳性率（召回率）
  - AUC = 1.0: 完美分类器
  - AUC = 0.5: 等同随机猜测
  - 适合类别平衡的场景

Precision-Recall曲线:
  - 横轴: 召回率（灵敏度）
  - 纵轴: 精确率
  - 适合类别不平衡的场景
  - Average Precision: PR曲线下的面积

多分类:
  - 使用One-vs-Rest或One-vs-One策略拆分为多个二分类问题
  - macro: 每个类别平等加权
  - weighted: 按类别样本数加权
""")
