"""
编码转换 —— 将分类特征转换为数值形式，适用于机器学习模型输入
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# ===================== 1. 构造示例数据 =====================
data = pd.DataFrame({
    "颜色": ["红", "蓝", "绿", "红", "蓝", "绿", "红", "蓝", "绿", "红"],
    "尺寸": ["S", "M", "L", "XL", "M", "S", "L", "XL", "M", "S"],
    "品牌": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
    "价格": [100, 200, 150, 300, 250, 120, 180, 280, 160, 220],
})
print("=== 原始数据 ===")
print(data)

# ===================== 2. 标签编码（LabelEncoder）=====================
# 将每个类别映射为一个整数，适用于有序分类或目标变量
le = LabelEncoder()
data["颜色_标签"] = le.fit_transform(data["颜色"])
print(f"\n=== 标签编码 ===")
print(f"映射关系: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(data[["颜色", "颜色_标签"]])

# 注意：标签编码会引入数值大小关系（0 < 1 < 2），对无序分类可能不合适

# ===================== 3. 有序编码（OrdinalEncoder）=====================
# 适用于有明确顺序的分类特征
# 注意：需要手动定义顺序，否则按字母序排列
size_order = [["S", "M", "L", "XL"]]
oe = OrdinalEncoder(categories=size_order)
data["尺寸_有序"] = oe.fit_transform(data[["尺寸"]])
print(f"\n=== 有序编码 ===")
print(f"顺序: S=0, M=1, L=2, XL=3")
print(data[["尺寸", "尺寸_有序"]])

# ===================== 4. 独热编码（OneHotEncoder）=====================
# 将每个类别转为独立的 0/1 列，适用于无序分类特征
ohe = OneHotEncoder(sparse_output=False, drop=None)  # drop="first" 可避免多重共线性
color_encoded = ohe.fit_transform(data[["颜色"]])
color_cols = ohe.get_feature_names_out(["颜色"])
data_ohe = pd.DataFrame(color_encoded, columns=color_cols)
print(f"\n=== 独热编码 ===")
print(f"类别: {ohe.categories_[0]}")
print(data_ohe)

# drop="first"：去掉第一列，避免虚拟变量陷阱（多重共线性）
ohe_drop = OneHotEncoder(sparse_output=False, drop="first")
color_drop = ohe_drop.fit_transform(data[["颜色"]])
color_drop_cols = ohe_drop.get_feature_names_out(["颜色"])
print(f"\n=== 独热编码（drop='first'）===")
print(pd.DataFrame(color_drop, columns=color_drop_cols))

# ===================== 5. Pandas get_dummies =====================
# 更方便的一键独热编码方式
data_dummies = pd.get_dummies(data[["颜色", "品牌"]], prefix=["颜色", "品牌"])
print(f"\n=== pd.get_dummies ===")
print(data_dummies)

# ===================== 6. 频率编码 =====================
# 用类别出现的频率替代类别值，适用于高基数特征
freq_map = data["品牌"].value_counts(normalize=True).to_dict()
data["品牌_频率"] = data["品牌"].map(freq_map)
print(f"\n=== 频率编码 ===")
print(f"频率映射: {freq_map}")
print(data[["品牌", "品牌_频率"]])

# ===================== 7. 目标编码（均值编码）=====================
# 用目标变量的条件均值替代类别值，注意：需防止过拟合（通常在训练集上计算，应用到测试集）
# 此处仅作演示，实际应先划分训练/测试集
target_mean = data.groupby("颜色")["价格"].mean().to_dict()
data["颜色_目标编码"] = data["颜色"].map(target_mean)
print(f"\n=== 目标编码 ===")
print(f"目标均值映射: {target_mean}")
print(data[["颜色", "价格", "颜色_目标编码"]])

# ===================== 8. 对比总结 =====================
print("\n=== 编码策略对比 ===")
print("1. 标签编码: 适合有序分类或目标变量，简单但引入虚假顺序")
print("2. 有序编码: 适合有明确顺序的特征（如 S<M<L<XL）")
print("3. 独热编码: 适合无序分类，维度随类别数增长，注意多重共线性")
print("4. 频率编码: 适合高基数特征，保留频率信息但丢失类别语义")
print("5. 目标编码: 信息量大但易过拟合，需交叉验证或正则化")
