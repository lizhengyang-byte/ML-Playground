"""
数据加载 (Data Loading)

各种数据源的加载方式和数据检查方法：
1. sklearn内置数据集
2. pandas读取CSV
3. numpy读取文本/数值文件
4. 数据基本信息检查
"""

import numpy as np
import pandas as pd

# ============================================================
# 1. sklearn内置数据集
# ============================================================
from sklearn.datasets import load_iris, load_wine, load_diabetes, make_classification

print("=" * 60)
print("1. sklearn内置数据集")
print("=" * 60)

# 经典Iris数据集
iris = load_iris()
print(f"Iris数据集:")
print(f"  类型: {type(iris)}")
print(f"  数据形状: {iris.data.shape}")
print(f"  标签形状: {iris.target.shape}")
print(f"  特征名: {list(iris.feature_names)}")
print(f"  目标名: {list(iris.target_names)}")
print(f"  样本数: {iris.data.shape[0]}, 特征数: {iris.data.shape[1]}")
print(f"  前3个样本:\n{iris.data[:3]}")
print(f"  前3个标签: {iris.target[:3]}")

# Wine数据集
wine = load_wine()
print(f"\nWine数据集:")
print(f"  数据形状: {wine.data.shape}")
print(f"  类别数: {len(wine.target_names)}")
print(f"  各类样本数: {dict(zip(wine.target_names, np.bincount(wine.target)))}")

# 回归数据集 - Diabetes
diabetes = load_diabetes()
print(f"\nDiabetes数据集 (回归):")
print(f"  数据形状: {diabetes.data.shape}")
print(f"  目标范围: [{diabetes.target.min():.1f}, {diabetes.target.max():.1f}]")

# 生成合成数据
X_synth, y_synth = make_classification(
    n_samples=100, n_features=5, n_informative=3, random_state=42
)
print(f"\n合成分类数据:")
print(f"  数据形状: {X_synth.shape}")
print(f"  标签分布: {dict(zip(*np.unique(y_synth, return_counts=True)))}")

# ============================================================
# 2. pandas读取CSV
# ============================================================
print("\n" + "=" * 60)
print("2. pandas读取CSV")
print("=" * 60)

# 创建示例CSV文件（模拟）
np.random.seed(42)
df_demo = pd.DataFrame({
    '年龄': np.random.randint(18, 65, 100),
    '收入': np.random.normal(50000, 15000, 100).astype(int),
    '学历': np.random.choice(['本科', '硕士', '博士'], 100),
    '城市': np.random.choice(['北京', '上海', '广州', '深圳'], 100),
    '是否购买': np.random.choice([0, 1], 100, p=[0.6, 0.4])
})

csv_path = 'sample_data.csv'
df_demo.to_csv(csv_path, index=False, encoding='utf-8-sig')

# 读取CSV
df_loaded = pd.read_csv(csv_path)
print(f"读取CSV成功:")
print(f"  形状: {df_loaded.shape}")
print(f"  列名: {list(df_loaded.columns)}")
print(f"  前5行:")
print(df_loaded.head().to_string(index=False))

# 基本信息
print(f"\n数据类型:")
print(df_loaded.dtypes)

# ============================================================
# 3. numpy读取文本/数值文件
# ============================================================
print("\n" + "=" * 60)
print("3. numpy读取文本/数值文件")
print("=" * 60)

# 保存numpy数组到文本文件
np.random.seed(42)
data_np = np.random.randn(50, 4)
np.savetxt('sample_numeric.txt', data_np, fmt='%.4f', delimiter=',')

# 读取
data_loaded = np.loadtxt('sample_numeric.txt', delimiter=',')
print(f"numpy读取文本文件:")
print(f"  形状: {data_loaded.shape}")
print(f"  前3行:\n{data_loaded[:3]}")

# 读取特定列
col0_col2 = np.loadtxt('sample_numeric.txt', delimiter=',', usecols=(0, 2))
print(f"\n  仅读取第0、2列: {col0_col2.shape}")
print(f"  前3行:\n{col0_col2[:3]}")

# ============================================================
# 4. 数据基本信息检查
# ============================================================
print("\n" + "=" * 60)
print("4. 数据基本信息检查")
print("=" * 60)

# 使用pandas检查
df_check = df_loaded.copy()

print(f"基本形状: {df_check.shape} ({df_check.shape[0]}行, {df_check.shape[1]}列)")
print(f"\n数值列统计:")
print(df_check.describe().to_string())

print(f"\n缺失值检查:")
missing = df_check.isnull().sum()
print(f"  各列缺失值: {dict(missing[missing > 0]) if missing.sum() > 0 else '无缺失值'}")

print(f"\n数据类型:")
for col in df_check.columns:
    print(f"  {col}: {df_check[col].dtype} ({'数值' if df_check[col].dtype in ['int64', 'float64'] else '分类'})")

# 分类列唯一值
print(f"\n分类列唯一值:")
for col in df_check.select_dtypes(include='object').columns:
    unique_vals = df_check[col].unique()
    print(f"  {col}: {list(unique_vals)} (共{len(unique_vals)}个)")

# 重复行检查
n_duplicates = df_check.duplicated().sum()
print(f"\n重复行数: {n_duplicates}")

# ============================================================
# 5. numpy数据检查
# ============================================================
print("\n" + "=" * 60)
print("5. numpy数组数据检查")
print("=" * 60)

arr = data_loaded
print(f"数组形状: {arr.shape}")
print(f"数据类型: {arr.dtype}")
print(f"内存占用: {arr.nbytes} 字节")
print(f"\n统计信息:")
print(f"  均值: {arr.mean(axis=0).round(4)}")
print(f"  标准差: {arr.std(axis=0).round(4)}")
print(f"  最小值: {arr.min(axis=0).round(4)}")
print(f"  最大值: {arr.max(axis=0).round(4)}")
print(f"  中位数: {np.median(arr, axis=0).round(4)}")

# 缺失值/无穷值检查
print(f"\nNaN数量: {np.isnan(arr).sum()}")
print(f"Inf数量: {np.isinf(arr).sum()}")

# 清理临时文件
import os
for f in [csv_path, 'sample_numeric.txt']:
    if os.path.exists(f):
        os.remove(f)
        print(f"\n已清理临时文件: {f}")
