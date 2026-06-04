"""
数据分割 —— 训练集/验证集/测试集的划分策略
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    ShuffleSplit,
    TimeSeriesSplit,
)

# ===================== 1. 生成示例数据 =====================
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=5,
    n_classes=3, n_clusters_per_class=1, random_state=42
)
print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"类别分布: {np.bincount(y)}")

# ===================== 2. 简单留出法（train_test_split）=====================
# 最基础的分割：按比例随机划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify 保持类别比例
)
print(f"\n=== 简单留出法 ===")
print(f"训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")
print(f"训练集类别分布: {np.bincount(y_train)}")
print(f"测试集类别分布: {np.bincount(y_test)}")

# 三层划分：训练/验证/测试
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)  # 0.25 × 0.8 = 0.2，所以最终 60/20/20
print(f"\n=== 三层划分 ===")
print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

# ===================== 3. K 折交叉验证（KFold）=====================
# 将数据均分为 K 份，每次取 1 份作验证，其余作训练
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"\n=== 5 折交叉验证 ===")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}: 训练={len(train_idx)}, 验证={len(val_idx)}")

# ===================== 4. 分层 K 折（StratifiedKFold）=====================
# 每折保持类别比例与整体一致，分类任务首选
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"\n=== 分层 5 折交叉验证 ===")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    train_dist = np.bincount(y[train_idx])
    val_dist = np.bincount(y[val_idx])
    print(f"Fold {fold+1}: 训练={len(train_idx)} {train_dist}, 验证={len(val_idx)} {val_dist}")

# ===================== 5. 留一法（LeaveOneOut）=====================
# 每次只留 1 个样本作验证，计算量大但偏差小
loo = LeaveOneOut()
print(f"\n=== 留一法 ===")
print(f"总折数: {loo.get_n_splits(X)}（等于样本数 {len(X)}）")
# 仅演示前 3 折
for i, (train_idx, val_idx) in enumerate(loo.split(X)):
    if i >= 3:
        break
    print(f"  Fold {i+1}: 训练={len(train_idx)}, 验证={len(val_idx)} (样本索引={val_idx[0]})")

# ===================== 6. 随机重复采样（ShuffleSplit）=====================
# 每次随机采样 train_size 样本作训练，剩余作验证，可重复多次
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
print(f"\n=== 随机重复采样 (5 次) ===")
for fold, (train_idx, val_idx) in enumerate(ss.split(X)):
    print(f"Split {fold+1}: 训练={len(train_idx)}, 验证={len(val_idx)}")

# ===================== 7. 时间序列分割（TimeSeriesSplit）=====================
# 保持时间顺序：训练集始终在验证集之前，防止未来信息泄露
tscv = TimeSeriesSplit(n_splits=5)
print(f"\n=== 时间序列分割 ===")
# 模拟时间序列数据
X_ts = np.arange(100).reshape(-1, 1)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_ts)):
    print(f"Fold {fold+1}: 训练=[0:{train_idx[-1]+1}]({len(train_idx)}), "
          f"验证=[{val_idx[0]}:{val_idx[-1]+1}]({len(val_idx)})")

# ===================== 8. 生成数据文件用于后续目录 =====================
# 保存一份干净的训练/测试集供后续算法使用
from sklearn.datasets import load_iris
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris)
print(f"\n=== Iris 数据集分割（供后续使用）===")
print(f"训练集: {X_tr.shape}, 测试集: {X_te.shape}")

print("\n=== 分割策略选择指南 ===")
print("1. 数据充足 → 简单留出法 (train_test_split)")
print("2. 数据有限 → K 折交叉验证 (StratifiedKFold)")
print("3. 分类任务 → 分层 K 折，保持类别平衡")
print("4. 时间序列 → TimeSeriesSplit，严禁打乱顺序")
print("5. 超参调优 → 用验证集/交叉验证；测试集只在最终评估时使用一次")
