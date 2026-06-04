"""
数据清洗 —— 处理重复值、异常值、数据类型转换等常见数据质量问题
"""
import numpy as np
import pandas as pd

# ===================== 1. 构造含脏数据的示例 =====================
np.random.seed(42)
n = 200
data = pd.DataFrame({
    "ID": list(range(1, n+1)),
    "姓名": np.random.choice(["张三", "李四", "王五", "赵六", "钱七"], n),
    "年龄": np.random.randint(15, 80, n).astype(float),
    "收入": np.random.normal(50000, 15000, n),
    "城市": np.random.choice(["北京", "上海", "广州", "深圳", "杭州"], n),
    "注册日期": pd.date_range("2020-01-01", periods=n, freq="D"),
})

# 注入脏数据
# 1) 重复行
duplicates = data.sample(10, random_state=42)
data = pd.concat([data, duplicates], ignore_index=True)

# 2) 异常值
data.loc[5, "年龄"] = -5          # 负数年龄
data.loc[10, "年龄"] = 200        # 不合理年龄
data.loc[15, "收入"] = -10000     # 负数收入
data.loc[20, "收入"] = 1e7        # 极端高收入

# 3) 空白/特殊字符
data.loc[25, "城市"] = "  北京 "  # 前后空格
data.loc[30, "城市"] = ""         # 空字符串
data.loc[35, "姓名"] = None       # 缺失

print("=== 原始脏数据（前 15 行）===")
print(data.head(15))
print(f"总行数: {len(data)}")

# ===================== 2. 处理重复值 =====================
dup_count = data.duplicated().sum()
print(f"\n=== 重复行数: {dup_count} ===")
data = data.drop_duplicates()
print(f"去重后行数: {len(data)}")

# ===================== 3. 清理空白和特殊字符 =====================
# 去除字符串列的首尾空白
for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].astype(str).str.strip()
    data[col] = data[col].replace("", np.nan)  # 空字符串转为 NaN

# 替换 "nan" 字符串（由 None 转化而来）
data = data.replace("nan", np.nan)
print(f"\n=== 清理空白后缺失情况 ===")
print(data.isnull().sum())

# ===================== 4. 处理异常值（基于 IQR）=====================
def detect_outliers_iqr(series, factor=1.5):
    """用 IQR 方法检测异常值，返回布尔掩码"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper), lower, upper

for col in ["年龄", "收入"]:
    mask, lower, upper = detect_outliers_iqr(data[col].dropna())
    outlier_count = mask.sum()
    print(f"\n{col}: IQR 范围 [{lower:.0f}, {upper:.0f}], 异常值 {outlier_count} 个")

    # 截断法（Winsorize）：将异常值限制在边界内
    data[col] = data[col].clip(lower=lower, upper=upper)

# ===================== 5. 基于 Z-score 的异常值检测 =====================
from scipy import stats

for col in ["年龄", "收入"]:
    z_scores = np.abs(stats.zscore(data[col].dropna()))
    outlier_z = (z_scores > 3).sum()
    print(f"{col}: Z-score>3 的异常值 {outlier_z} 个")

# ===================== 6. 修正数据范围 =====================
# 年龄应在合理范围内
data["年龄"] = data["年龄"].clip(lower=0, upper=120)
# 收入不应为负
data["收入"] = data["收入"].clip(lower=0)
print(f"\n=== 修正后的数据范围 ===")
print(f"年龄: [{data['年龄'].min():.0f}, {data['年龄'].max():.0f}]")
print(f"收入: [{data['收入'].min():.0f}, {data['收入'].max():.0f}]")

# ===================== 7. 数据类型转换 =====================
# 将年龄转为整数
data["年龄"] = data["年龄"].astype(int)
# 日期列确保为 datetime 类型
data["注册日期"] = pd.to_datetime(data["注册日期"], errors="coerce")
print(f"\n=== 数据类型 ===")
print(data.dtypes)

# ===================== 8. 最终清洗结果 =====================
print(f"\n=== 清洗后数据（前 10 行）===")
print(data.head(10))
print(f"最终行数: {len(data)}, 缺失数: {data.isnull().sum().sum()}")

print("\n=== 数据清洗流程总结 ===")
print("1. 去除重复行")
print("2. 清理空白字符、统一格式")
print("3. 检测并处理异常值（IQR / Z-score）")
print("4. 修正数据范围（clip 到合理区间）")
print("5. 数据类型转换（确保正确类型）")
print("6. 处理缺失值（参见 缺失值处理.py）")
