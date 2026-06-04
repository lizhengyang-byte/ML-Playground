"""
特征提取 (Feature Extraction)

从原始数据中构造新的特征表示，包括：
1. PCA 主成分分析 - 线性降维，找到方差最大的投影方向
2. 多项式特征 (PolynomialFeatures) - 生成特征的高次项和交叉项
3. 文本特征 (TfidfVectorizer) - 将文本转换为TF-IDF数值向量
4. 图像块特征 (PatchExtractor) - 从图像中提取局部像素块作为特征
"""

import numpy as np
from sklearn.datasets import make_classification

# ============================================================
# 1. PCA 主成分分析
# ============================================================
from sklearn.decomposition import PCA

print("=" * 60)
print("1. PCA 主成分分析")
print("=" * 60)

# 生成高维数据（10个特征）
X_pca, y_pca = make_classification(
    n_samples=200, n_features=10, n_informative=3,
    n_redundant=2, random_state=42
)

print(f"原始数据维度: {X_pca.shape}")

# 保留主成分能解释95%方差
pca_auto = PCA(n_components=0.95, random_state=42)
X_reduced = pca_auto.fit_transform(X_pca)

print(f"保留95%方差后维度: {X_reduced.shape}")
print(f"各主成分方差贡献率: {np.round(pca_auto.explained_variance_ratio_, 4)}")
print(f"累计方差贡献率:     {np.round(np.cumsum(pca_auto.explained_variance_ratio_), 4)}")
print(f"主成分数量: {pca_auto.n_components_} / 原始: {X_pca.shape[1]}")

# 指定保留主成分数量
pca_3 = PCA(n_components=3, random_state=42)
X_3d = pca_3.fit_transform(X_pca)
print(f"\n指定保留3个主成分: {X_3d.shape}")
print(f"3个主成分方差贡献率: {np.round(pca_3.explained_variance_ratio_, 4)}")
print(f"3个主成分累计贡献率: {pca_3.explained_variance_ratio_.sum():.4f}")

# ============================================================
# 2. 多项式特征 (PolynomialFeatures)
# ============================================================
from sklearn.preprocessing import PolynomialFeatures

print("\n" + "=" * 60)
print("2. 多项式特征 (PolynomialFeatures)")
print("=" * 60)

# 简单的2特征示例，便于理解
X_poly_demo = np.array([[2, 3],
                         [4, 5],
                         [6, 1]])
print(f"原始数据 (2个特征):\n{X_poly_demo}")

# degree=2, interaction_only=False: 包含平方项和交叉项
poly_full = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly_full = poly_full.fit_transform(X_poly_demo)
print(f"\ndegree=2 (含平方项和交叉项):")
print(f"特征名: {poly_full.get_feature_names_out(['x1', 'x2'])}")
print(f"形状: {X_poly_full.shape}")
print(f"数据:\n{X_poly_full}")

# interaction_only=True: 只有交叉项，无平方项
poly_interact = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly_inter = poly_interact.fit_transform(X_poly_demo)
print(f"\ninteraction_only=True (仅交叉项):")
print(f"特征名: {poly_interact.get_feature_names_out(['x1', 'x2'])}")
print(f"数据:\n{X_poly_inter}")

# ============================================================
# 3. 文本特征 (TfidfVectorizer)
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer

print("\n" + "=" * 60)
print("3. 文本特征 (TfidfVectorizer)")
print("=" * 60)

# 示例文档集
corpus = [
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络模型",
    "支持向量机是一种监督学习算法",
    "自然语言处理用于分析文本数据",
    "卷积神经网络在图像识别中效果很好",
]

print("原始文档:")
for i, doc in enumerate(corpus):
    print(f"  [{i}] {doc}")

# TF-IDF向量化
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)

print(f"\nTF-IDF矩阵形状: {X_tfidf.shape} (文档数 x 词汇量)")
print(f"词汇表: {tfidf.get_feature_names_out()[:10]}... (共{len(tfidf.get_feature_names_out())}个)")

# 以稀疏矩阵形式查看
X_dense = X_tfidf.toarray()
print(f"\nTF-IDF矩阵 (稠密形式，保留3位小数):")
print(f"{'词汇':<12}", end="")
for i in range(min(8, X_dense.shape[1])):
    print(f"{tfidf.get_feature_names_out()[i]:<10}", end="")
print("...")
for doc_idx in range(len(corpus)):
    print(f"文档{doc_idx}:  ", end="")
    for feat_idx in range(min(8, X_dense.shape[1])):
        print(f"{X_dense[doc_idx, feat_idx]:<10.3f}", end="")
    print("...")

# IDF权重：衡量词的区分能力
print("\nIDF权重 (高=更独特):")
idf = tfidf.idf_
top_idf_idx = np.argsort(idf)[::-1][:8]
for idx in top_idf_idx:
    word = tfidf.get_feature_names_out()[idx]
    print(f"  {word}: {idf[idx]:.4f}")

# ============================================================
# 4. 图像块特征 (PatchExtractor)
# ============================================================
from sklearn.feature_extraction.image import PatchExtractor

print("\n" + "=" * 60)
print("4. 图像块特征 (PatchExtractor)")
print("=" * 60)

# 模拟一张灰度图像 (28x28)
np.random.seed(42)
fake_image = np.random.rand(1, 28, 28)
print(f"模拟图像形状: {fake_image.shape} (1张, 28x28像素)")

# 提取8x8的图像块
patch_extractor = PatchExtractor(patch_size=(8, 8), random_state=42)
patches = patch_extractor.transform(fake_image)

print(f"提取的块数量: {patches.shape[0]}")
print(f"每个块的大小: {patches.shape[1]}x{patches.shape[2]}")
print(f"展平后每个块的特征维度: {patches.shape[1] * patches.shape[2]}")

# 查看一个块的统计信息
patch_0 = patches[0]
print(f"\n第1个块的统计:")
print(f"  最小值: {patch_0.min():.4f}, 最大值: {patch_0.max():.4f}")
print(f"  均值: {patch_0.mean():.4f}, 标准差: {patch_0.std():.4f}")

# 多张图像的块提取
images = np.random.rand(5, 32, 32)
patches_multi = patch_extractor.transform(images)
print(f"\n5张32x32图像提取8x8块后:")
print(f"  总块数: {patches_multi.shape[0]}")
print(f"  每个块: {patches_multi.shape[1]}x{patches_multi.shape[2]}")

# ============================================================
# 综合对比
# ============================================================
print("\n" + "=" * 60)
print("综合对比：特征提取方法")
print("=" * 60)
print(f"PCA:           {X_pca.shape[1]}维 -> {X_reduced.shape[1]}维 (保留95%方差)")
print(f"多项式特征:    2维 -> {X_poly_full.shape[1]}维 (degree=2)")
print(f"TF-IDF文本:    {len(corpus)}篇文档 -> {X_tfidf.shape[1]}维词汇空间")
print(f"图像块特征:    28x28 -> {patches.shape[0]}个8x8块")
