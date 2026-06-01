"""
自训练（Self-Training）—— 半监督学习：用少量有标签数据训练模型，
将高置信度的无标签预测作为伪标签加入训练集，迭代训练
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.semi_supervised import SelfTrainingClassifier

# ===================== 1. 加载并准备数据 =====================
iris = load_iris()
X, y = iris.data, iris.target
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 只保留少量标签（约 10%），其余标记为 -1（未标记）
np.random.seed(42)
n_labeled = int(len(y_train_full) * 0.1)  # 仅 10% 有标签
labeled_idx = np.random.choice(len(y_train_full), n_labeled, replace=False)
y_train_semi = np.full_like(y_train_full, fill_value=-1)
y_train_semi[labeled_idx] = y_train_full[labeled_idx]

print(f"=== 数据划分 ===")
print(f"训练集: {len(X_train_full)} 样本 (有标签: {(y_train_semi >= 0).sum()}, "
      f"无标签: {(y_train_semi == -1).sum()})")
print(f"测试集: {len(X_test)} 样本")

# ===================== 2. 有标签数据训练（基准）=====================
lr_labeled = RandomForestClassifier(n_estimators=100, random_state=42)
lr_labeled.fit(X_train_full[labeled_idx], y_train_full[labeled_idx])
acc_labeled = lr_labeled.score(X_test, y_test)
print(f"\n=== 仅用有标签数据训练 ===")
print(f"测试准确率: {acc_labeled:.4f}")

# ===================== 3. 全部标签训练（上限）=====================
lr_full = RandomForestClassifier(n_estimators=100, random_state=42)
lr_full.fit(X_train_full, y_train_full)
acc_full = lr_full.score(X_test, y_test)
print(f"\n=== 全部标签训练（理想上限）===")
print(f"测试准确率: {acc_full:.4f}")

# ===================== 4. SelfTrainingClassifier =====================
# threshold: 伪标签的最低置信度阈值
# max_iter: 最大迭代次数
# criterion: 选择伪标签的策略 ("threshold" 或 "k_best")
st = SelfTrainingClassifier(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold=0.8,
    max_iter=20,
    criterion="threshold",
)
st.fit(X_train_full, y_train_semi)

acc_semi = st.score(X_test, y_test)
print(f"\n=== 自训练 (Self-Training) ===")
print(f"测试准确率: {acc_semi:.4f}")
print(f"迭代次数: {st.n_iter_}")
print(f"最终有标签样本数: {len(st.labeled_iter_)}")
# labeled_iter_ 记录每轮新增的伪标签

# ===================== 5. 不同置信度阈值 =====================
print("\n=== 不同阈值 (threshold) 对比 ===")
for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    st_t = SelfTrainingClassifier(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        threshold=thresh, max_iter=20,
    )
    st_t.fit(X_train_full, y_train_semi)
    acc = st_t.score(X_test, y_test)
    print(f"  threshold={thresh}: 测试准确率={acc:.4f}, 迭代={st_t.n_iter_}")

# ===================== 6. 手动实现简单自训练 =====================
print("\n=== 手动自训练流程 ===")
base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
X_labeled = X_train_full[labeled_idx].copy()
y_labeled = y_train_full[labeled_idx].copy()
X_unlabeled = np.delete(X_train_full, labeled_idx, axis=0)
y_unlabeled_true = np.delete(y_train_full, labeled_idx)

for iteration in range(5):
    # 1. 用当前有标签数据训练
    base_clf.fit(X_labeled, y_labeled)
    # 2. 对无标签数据预测
    probs = base_clf.predict_proba(X_unlabeled)
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    # 3. 选取高置信度预测作为伪标签
    high_conf = max_probs >= 0.9
    if high_conf.sum() == 0:
        print(f"  迭代 {iteration+1}: 无高置信度样本，停止")
        break
    # 4. 加入训练集
    X_labeled = np.vstack([X_labeled, X_unlabeled[high_conf]])
    y_labeled = np.concatenate([y_labeled, preds[high_conf]])
    X_unlabeled = X_unlabeled[~high_conf]
    print(f"  迭代 {iteration+1}: 新增 {high_conf.sum()} 个伪标签, "
          f"总标签数={len(y_labeled)}, 剩余无标签={len(X_unlabeled)}")

# 最终评估
acc_manual = base_clf.score(X_test, y_test)
print(f"  手动自训练测试准确率: {acc_manual:.4f}")

print("\n=== 自训练要点 ===")
print("- 优点：简单直观，不需要特殊架构")
print("- 缺点：初始模型差时，伪标签错误会传播（确认偏差）")
print("- threshold 太低：引入错误标签；太高：几乎不新增样本")
print("- 适合：初始模型已有一定准确率的场景")
print("- 与全监督对比，半监督可显著提升标签不足时的性能")
