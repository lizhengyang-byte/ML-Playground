"""
随机种子与可复现性 (Random Seeds & Reproducibility)

机器学习实验的可复现性是科学研究和工程实践的基础：
1. 各框架的随机种子设置
2. 全局可复现性封装
3. 验证可复现性
4. 常见陷阱与注意事项
"""

import os
import random
import numpy as np

# ============================================================
# 1. 各框架随机种子设置
# ============================================================
print("=" * 60)
print("1. 各框架随机种子设置")
print("=" * 60)

SEED = 42

# Python 内置 random
random.seed(SEED)
print(f"random.seed({SEED}) 已设置")

# NumPy
np.random.seed(SEED)
print(f"np.random.seed({SEED}) 已设置")

# PyTorch (CPU)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"torch.manual_seed({SEED}) 已设置")
    if torch.cuda.is_available():
        print(f"  CUDA: deterministic=True, benchmark=False")
except ImportError:
    print("PyTorch 未安装，跳过")

# scikit-learn
print(f"sklearn: 通过 random_state 参数控制 (非全局种子)")

print(f"\n要点: 每个框架有独立的随机状态，需分别设置")

# ============================================================
# 2. 验证可复现性 —— NumPy
# ============================================================
print("\n" + "=" * 60)
print("2. 验证可复现性 —— NumPy")
print("=" * 60)

# 第一次生成
np.random.seed(SEED)
arr1 = np.random.randn(5)

# 第二次生成
np.random.seed(SEED)
arr2 = np.random.randn(5)

print(f"第1次: {arr1.round(4)}")
print(f"第2次: {arr2.round(4)}")
print(f"结果一致: {np.array_equal(arr1, arr2)}")

# ============================================================
# 3. 验证可复现性 —— PyTorch
# ============================================================
print("\n" + "=" * 60)
print("3. 验证可复现性 —— PyTorch")
print("=" * 60)

try:
    import torch
    import torch.nn as nn

    # 第一次
    torch.manual_seed(SEED)
    t1 = torch.randn(5)

    # 第二次
    torch.manual_seed(SEED)
    t2 = torch.randn(5)

    print(f"第1次: {t1.numpy().round(4)}")
    print(f"第2次: {t2.numpy().round(4)}")
    print(f"结果一致: {torch.equal(t1, t2)}")
except ImportError:
    print("PyTorch 未安装，跳过")

# ============================================================
# 4. 全局可复现性封装函数
# ============================================================
print("\n" + "=" * 60)
print("4. 全局可复现性封装函数")
print("=" * 60)

def set_seed(seed=42):
    """设置所有框架的随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        from sklearn.utils import check_random_state
        # sklearn 通过 random_state 参数使用，无需全局设置
    except ImportError:
        pass

    print(f"全局种子已设置: {seed}")

set_seed(SEED)

# ============================================================
# 5. 验证封装函数
# ============================================================
print("\n" + "=" * 60)
print("5. 验证封装函数 —— 多次调用结果一致")
print("=" * 60)

results = []
for run in range(3):
    set_seed(SEED)
    a = np.random.randn(3)
    b = np.random.randint(0, 100, 3)
    results.append((a.round(4), b))

for i, (a, b) in enumerate(results):
    print(f"  Run {i+1}: arr={a}, int={b}")

all_same_a = all(np.array_equal(r[0], results[0][0]) for r in results)
all_same_b = all(np.array_equal(r[1], results[0][1]) for r in results)
print(f"\n  数组一致: {all_same_a}, 整数一致: {all_same_b}")

# ============================================================
# 6. sklearn 模型可复现性
# ============================================================
print("\n" + "=" * 60)
print("6. sklearn 模型可复现性")
print("=" * 60)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=SEED
)

# 两次训练结果一致
accs = []
for run in range(2):
    clf = RandomForestClassifier(n_estimators=50, random_state=SEED)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    accs.append(acc)
    print(f"  Run {run+1}: 准确率={acc:.6f}")

print(f"  两次结果一致: {accs[0] == accs[1]}")

# ============================================================
# 7. 常见陷阱
# ============================================================
print("\n" + "=" * 60)
print("7. 常见陷阱与注意事项")
print("=" * 60)

print("""
陷阱1: 忘记设置某个框架的种子
  错误: 只设了 np.random.seed, 忘了 torch.manual_seed
  后果: PyTorch部分不可复现

陷阱2: 多进程/并行时种子不一致
  原因: 每个worker有独立的随机状态
  解决: 为每个worker设置不同的种子 (base_seed + worker_id)

陷阱3: GPU计算的非确定性
  原因: cuDNN的benchmark和浮点运算顺序
  解决: torch.backends.cudnn.deterministic = True
  代价: 可能降低训练速度

陷阱4: 数据加载顺序
  原因: DataLoader的shuffle和num_workers
  解决: 设置worker_init_fn, 固定shuffle顺序

陷阱5: 版本差异
  不同版本的库可能产生不同结果
  建议: 记录所有依赖版本 (pip freeze)
""")

# ============================================================
# 8. 实用技巧
# ============================================================
print("=" * 60)
print("8. 实用技巧")
print("=" * 60)

print("1. 项目根目录创建 seed_utils.py, 所有实验导入使用")
print("2. 命令行传入 --seed 参数, 方便对比不同种子")
print("3. 记录种子到日志/配置文件, 方便追溯")
print("4. 关键实验用多个种子运行, 报告均值±标准差")
print("5. 部署时关闭 determinist ic 模式以提升性能")
