"""
特征交叉 (Feature Crossing)

通过组合原始特征构造新特征，增强模型的表达能力：
1. PolynomialFeatures - 自动生成多项式和交叉特征
2. 手动构造交互特征 - 根据业务理解手动组合
3. 分类特征交叉 - 类别之间的组合特征
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# ============================================================
# 1. PolynomialFeatures 自动生成交叉特征
# ============================================================
from sklearn.preprocessing import PolynomialFeatures

print("=" * 60)
print("1. PolynomialFeatures 自动生成交叉特征")
print("=" * 60)

# 简单数据示例
X_demo = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
])
feature_names_base = ['面积', '房龄']

print(f"原始数据:\n{X_demo}")
print(f"特征: {feature_names_base}")

# degree=2, include_bias=False
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_demo)

print(f"\ndegree=2 多项式特征:")
feature_names_poly = poly.get_feature_names_out(feature_names_base)
for i, name in enumerate(feature_names_poly):
    print(f"  [{i}] {name}: {X_poly[:, i]}")

# 交叉项（不含平方项）
poly_interact = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_interact = poly_interact.fit_transform(X_demo)
feature_names_interact = poly_interact.get_feature_names_out(feature_names_base)

print(f"\n仅交叉项 (interaction_only=True):")
for i, name in enumerate(feature_names_interact):
    print(f"  [{i}] {name}: {X_interact[:, i]}")

# ============================================================
# 2. 手动构造交互特征
# ============================================================
print("\n" + "=" * 60)
print("2. 手动构造交互特征")
print("=" * 60)

np.random.seed(42)
n = 100

# 模拟房价数据
area = np.random.uniform(50, 200, n)       # 面积
rooms = np.random.randint(1, 6, n)          # 房间数
age = np.random.uniform(0, 30, n)           # 房龄
district = np.random.choice(['A', 'B', 'C'], n)  # 区域

price = 5000 * area + 100000 * rooms - 3000 * age + np.random.normal(0, 50000, n)

df = pd.DataFrame({
    '面积': area,
    '房间数': rooms,
    '房龄': age,
    '区域': district,
    '房价': price
})

print("原始数据前5行:")
print(df.head().to_string(index=False))

# 交互特征1: 面积 / 房间数 (每间房面积)
df['每间房面积'] = df['面积'] / df['房间数']

# 交互特征2: 面积 * 房龄 (面积的老化效应)
df['面积房龄交互'] = df['面积'] * df['房龄']

# 交互特征3: 面积^2 (面积的非线性效应)
df['面积平方'] = df['面积'] ** 2

# 交互特征4: 房间数与房龄的比值
df['房间房龄比'] = df['房间数'] / (df['房龄'] + 1)

print("\n构造交互特征后:")
print(df[['面积', '房间数', '房龄', '每间房面积', '面积房龄交互', '面积平方', '房间房龄比']].head().to_string(index=False))

# ============================================================
# 3. 分类特征交叉
# ============================================================
print("\n" + "=" * 60)
print("3. 分类特征交叉")
print("=" * 60)

np.random.seed(42)
n2 = 200

gender = np.random.choice(['男', '女'], n2)
city = np.random.choice(['北京', '上海', '广州'], n2)
education = np.random.choice(['本科', '硕士', '博士'], n2)

# 方法1: 字符串拼接
df_cross = pd.DataFrame({
    '性别': gender,
    '城市': city,
    '学历': education,
})

# 分类交叉: 性别 + 城市
df_cross['性别_城市'] = df_cross['性别'] + '_' + df_cross['城市']

# 分类交叉: 城市 + 学历
df_cross['城市_学历'] = df_cross['城市'] + '_' + df_cross['学历']

# 三重交叉: 性别 + 城市 + 学历
df_cross['性别_城市_学历'] = df_cross['性别'] + '_' + df_cross['城市'] + '_' + df_cross['学历']

print("分类特征交叉:")
print(df_cross.head(10).to_string(index=False))

# 各组合的数量统计
print(f"\n性别_城市组合分布:")
print(df_cross['性别_城市'].value_counts().sort_index().to_string())

print(f"\n城市_学历组合分布:")
print(df_cross['城市_学历'].value_counts().sort_index().to_string())

# 方法2: 使用pd.get_dummies对交叉特征做独热编码
print("\n" + "=" * 60)
print("4. 交叉特征的独热编码")
print("=" * 60)

# 先独热编码再手动交叉
df_dummies = pd.get_dummies(df_cross[['性别', '城市']], prefix=['性别', '城市'])
print(f"独热编码后维度: {df_dummies.shape}")
print(df_dummies.head().to_string(index=False))

# 手动交叉独热编码列
cross_features = pd.DataFrame()
for g_col in [c for c in df_dummies.columns if c.startswith('性别_')]:
    for c_col in [c for c in df_dummies.columns if c.startswith('城市_')]:
        cross_name = f"{g_col}_X_{c_col}"
        cross_features[cross_name] = df_dummies[g_col] * df_dummies[c_col]

print(f"\n交叉后的独热编码维度: {cross_features.shape}")
print(cross_features.head().to_string(index=False))

# ============================================================
# 综合总结
# ============================================================
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("特征交叉的核心思想:")
print("  1. PolynomialFeatures: 自动化，适合数值特征，注意维度爆炸")
print("  2. 手动交互特征: 业务驱动，可解释性强，需要领域知识")
print("  3. 分类特征交叉: 捕捉类别间的联合效应，注意组合数爆炸")
print("  4. 交叉后的独热编码: 维度高，适合稀疏模型或树模型")
