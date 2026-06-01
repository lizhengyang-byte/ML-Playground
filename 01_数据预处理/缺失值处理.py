"""
缺失值处理 —— 展示常见的缺失值检测与填充/删除策略
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ===================== 1. 构造含缺失值的示例数据 =====================
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "年龄": np.random.randint(18, 70, n).astype(float),
    "收入": np.random.normal(50000, 15000, n),
    "学历": np.random.choice(["高中", "本科", "硕士", "博士"], n),
    "信用分": np.random.uniform(300, 850, n),
})

# 随机注入缺失值（约 10%）
for col in ["年龄", "收入", "信用分"]:
    mask = np.random.rand(n) < 0.10
    data.loc[mask, col] = np.nan

# 学历也注入少量缺失
mask_edu = np.random.rand(n) < 0.05
data.loc[mask_edu, "学历"] = np.nan

print("=== 原始数据（前 10 行）===")
print(data.head(10))
print(f"\n各列缺失数量:\n{data.isnull().sum()}")
print(f"总缺失率: {data.isnull().mean().mean():.2%}")

# ===================== 2. 删除策略 =====================
# 2a. 删除含缺失值的行（dropna）
data_dropped = data.dropna()
print(f"\n=== dropna 后剩余行数: {len(data_dropped)} (原始 {len(data)}) ===")

# 2b. 设定阈值：删除缺失超过 30% 列的行
thresh = int(data.shape[1] * 0.7)  # 至少 70% 非空
data_thresh = data.dropna(thresh=thresh)
print(f"thresh={thresh} 后剩余行数: {len(data_thresh)}")

# ===================== 3. 均值/中位数/众数填充 =====================
data_num = data[["年龄", "收入", "信用分"]]

# 均值填充
imputer_mean = SimpleImputer(strategy="mean")
filled_mean = pd.DataFrame(imputer_mean.fit_transform(data_num), columns=data_num.columns)
print(f"\n=== 均值填充后缺失: {filled_mean.isnull().sum().sum()} ===")

# 中位数填充
imputer_median = SimpleImputer(strategy="median")
filled_median = pd.DataFrame(imputer_median.fit_transform(data_num), columns=data_num.columns)
print(f"中位数填充后缺失: {filled_median.isnull().sum().sum()}")

# 众数填充（适用于分类特征）
imputer_mode = SimpleImputer(strategy="most_frequent")
edu_filled = imputer_mode.fit_transform(data[["学历"]])
print(f"学历众数填充后缺失: {pd.DataFrame(edu_filled).isnull().sum().sum()}")

# ===================== 4. 常量填充 =====================
imputer_const = SimpleImputer(strategy="constant", fill_value=0)
filled_const = pd.DataFrame(imputer_const.fit_transform(data_num), columns=data_num.columns)
print(f"\n=== 常量(0)填充后缺失: {filled_const.isnull().sum().sum()} ===")

# ===================== 5. KNN 填充 =====================
# 利用 K 个最近邻样本的加权平均值填充，考虑特征间相关性
imputer_knn = KNNImputer(n_neighbors=5, weights="distance")
filled_knn = pd.DataFrame(imputer_knn.fit_transform(data_num), columns=data_num.columns)
print(f"\n=== KNN 填充后缺失: {filled_knn.isnull().sum().sum()} ===")
print("KNN 填充后的统计描述:")
print(filled_knn.describe().round(2))

# ===================== 6. 多重插补（IterativeImputer）=====================
# 模拟 R 的 MICE：对每个特征建立回归模型，迭代填充
imputer_iter = IterativeImputer(max_iter=10, random_state=42)
filled_iter = pd.DataFrame(imputer_iter.fit_transform(data_num), columns=data_num.columns)
print(f"\n=== 迭代插补后缺失: {filled_iter.isnull().sum().sum()} ===")
print("迭代插补后统计描述:")
print(filled_iter.describe().round(2))

# ===================== 7. 对比各策略的填充效果 =====================
print("\n=== 各策略填充后的均值对比 ===")
original_mean = data_num.mean()
print(f"原始均值:\n{original_mean.round(2)}\n")
for name, filled in [("均值", filled_mean), ("中位数", filled_median),
                      ("常量(0)", filled_const), ("KNN", filled_knn), ("迭代", filled_iter)]:
    print(f"{name}填充均值:\n{filled.mean().round(2)}\n")
