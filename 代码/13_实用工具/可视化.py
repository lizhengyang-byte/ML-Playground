"""
可视化工具 (Visualization Utilities)

文本输出方式的数据检查和摘要统计（不依赖matplotlib/seaborn）：
1. 数据表格展示
2. 文本直方图
3. 相关性矩阵
4. 分布统计
"""

import numpy as np
import pandas as pd

# ============================================================
# 1. 数据表格展示
# ============================================================
print("=" * 60)
print("1. 数据表格展示")
print("=" * 60)

np.random.seed(42)
n = 200
df = pd.DataFrame({
    '年龄': np.random.randint(18, 65, n),
    '收入': np.random.normal(50000, 15000, n).astype(int),
    '消费': np.random.normal(5000, 2000, n).round(2),
    '满意度': np.random.choice(['低', '中', '高'], n, p=[0.2, 0.5, 0.3]),
    '是否会员': np.random.choice([0, 1], n, p=[0.4, 0.6])
})

print(f"数据集概览:")
print(f"  总行数: {len(df)}")
print(f"  总列数: {len(df.columns)}")
print(f"\n前10行数据:")
print(df.head(10).to_string(index=True))

# ============================================================
# 2. 文本直方图
# ============================================================
print("\n" + "=" * 60)
print("2. 文本直方图")
print("=" * 60)

def text_histogram(data, bins=10, width=40):
    """用文本绘制直方图"""
    counts, edges = np.histogram(data, bins=bins)
    max_count = counts.max()
    labels = []

    for i in range(len(counts)):
        low, high = edges[i], edges[i+1]
        bar_len = int(counts[i] / max_count * width) if max_count > 0 else 0
        bar = '#' * bar_len
        label = f"  [{low:>8.1f}, {high:>8.1f}) |{bar:<{width}} {counts[i]:>4}"
        labels.append(label)

    return '\n'.join(labels)

print("年龄分布:")
print(text_histogram(df['年龄'], bins=8))
print(f"\n年龄统计:")
print(f"  均值: {df['年龄'].mean():.1f}, 中位数: {df['年龄'].median():.1f}")
print(f"  标准差: {df['年龄'].std():.1f}")
print(f"  范围: [{df['年龄'].min()}, {df['年龄'].max()}]")

print("\n收入分布:")
print(text_histogram(df['收入'], bins=8))

# ============================================================
# 3. 分类变量频率表
# ============================================================
print("\n" + "=" * 60)
print("3. 分类变量频率表")
print("=" * 60)

def text_bar(value_counts, width=30):
    """用文本绘制水平条形图"""
    max_val = value_counts.max()
    lines = []
    for label, count in value_counts.items():
        bar_len = int(count / max_val * width) if max_val > 0 else 0
        bar = '#' * bar_len
        pct = count / value_counts.sum() * 100
        lines.append(f"  {str(label):<8} |{bar:<{width}} {count:>4} ({pct:>5.1f}%)")
    return '\n'.join(lines)

print("满意度分布:")
print(text_bar(df['满意度'].value_counts().sort_index()))

print("\n是否会员分布:")
print(text_bar(df['是否会员'].value_counts().sort_index()))

# ============================================================
# 4. 相关性矩阵 (文本展示)
# ============================================================
print("\n" + "=" * 60)
print("4. 相关性矩阵")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()

# 文本形式展示相关性
print(f"{'':>10}", end="")
for col in numeric_cols:
    print(f"{col:>10}", end="")
print()

for i, row_name in enumerate(numeric_cols):
    print(f"{row_name:>10}", end="")
    for j, col_name in enumerate(numeric_cols):
        val = corr.iloc[i, j]
        # 用符号表示相关性强度
        if abs(val) > 0.7:
            marker = "*"
        elif abs(val) > 0.3:
            marker = "+"
        else:
            marker = " "
        print(f"{val:>9.3f}{marker}", end="")
    print()

print("\n图例: *=强相关(>0.7), +=中等相关(>0.3)")

# ============================================================
# 5. 箱线图文本表示
# ============================================================
print("\n" + "=" * 60)
print("5. 箱线图文本表示")
print("=" * 60)

def text_boxplot(data, name=""):
    """文本形式展示箱线图信息"""
    q1 = np.percentile(data, 25)
    q2 = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_fence = max(data.min(), q1 - 1.5 * iqr)
    upper_fence = min(data.max(), q3 + 1.5 * iqr)
    outliers = np.sum((data < lower_fence) | (data > upper_fence))

    # 简单的文本箱线图
    min_val, max_val = data.min(), data.max()
    scale = 50  # 字符宽度
    rng = max_val - min_val if max_val > min_val else 1

    def pos(val):
        return int((val - min_val) / rng * scale)

    line = [' '] * (scale + 1)
    # 画箱体
    for i in range(pos(q1), pos(q3) + 1):
        line[i] = '-'
    line[pos(q2)] = '|'

    print(f"\n  {name}:")
    print(f"  Min={min_val:.2f}  Q1={q1:.2f}  Median={q2:.2f}  Q3={q3:.2f}  Max={max_val:.2f}")
    print(f"  IQR={iqr:.2f}, 异常值数={outliers}")
    print(f"  [{''.join(line)}]")

for col in ['年龄', '收入', '消费']:
    text_boxplot(df[col].values, col)

# ============================================================
# 6. 交叉频率表
# ============================================================
print("\n" + "=" * 60)
print("6. 交叉频率表 (分类变量)")
print("=" * 60)

# 满意度 vs 是否会员
cross = pd.crosstab(df['满意度'], df['是否会员'], margins=True)
cross.columns = ['非会员', '会员', '合计']
cross.index.name = '满意度'

print(cross.to_string())

print("\n会员比例 (按满意度):")
for satisfaction in ['低', '中', '高']:
    subset = df[df['满意度'] == satisfaction]
    member_rate = subset['是否会员'].mean() * 100
    bar_len = int(member_rate / 100 * 20)
    bar = '#' * bar_len
    print(f"  {satisfaction}: {bar} {member_rate:.1f}%")

# ============================================================
# 7. 数据摘要输出
# ============================================================
print("\n" + "=" * 60)
print("7. 综合数据摘要")
print("=" * 60)

print(f"数据集: {df.shape[0]}行 x {df.shape[1]}列")
print(f"\n数值特征摘要:")
for col in df.select_dtypes(include=[np.number]).columns:
    series = df[col]
    print(f"  {col}:")
    print(f"    均值={series.mean():.2f}, 中位数={series.median():.2f}, "
          f"std={series.std():.2f}, 范围=[{series.min()}, {series.max()}]")

print(f"\n分类特征摘要:")
for col in df.select_dtypes(include='object').columns:
    vc = df[col].value_counts()
    top = vc.index[0]
    print(f"  {col}: {len(vc)}个类别, 最频繁=\"{top}\"({vc.iloc[0]}次, {vc.iloc[0]/len(df)*100:.1f}%)")
