# SVD 奇异值分解：矩阵的"基因组"

> 所属模块：03_无监督学习/降维 | 源文件：SVD.py | 核心功能：完整 SVD 分解、截断 SVD、推荐系统模拟、文本 LSA

## 概述

SVD（Singular Value Decomposition）是线性代数中最重要的矩阵分解之一：X = U × Σ × Vᵀ。它把任意矩阵分解为三个矩阵的乘积——U 是左奇异向量，Σ 是奇异值（按降序排列），Vᵀ 是右奇异向量。

SVD 的应用极其广泛：PCA 的底层就是 SVD、推荐系统的核心是矩阵分解、图像压缩的本质是截断 SVD、文本处理的潜在语义分析（LSA）也是 SVD。

脚本演示了完整 SVD 分解、截断降维、推荐系统模拟和文本 LSA 四个应用场景。

## 关键代码解释

### 奇异值的能量分布

`python
U, S, Vt = svd(X_scaled, full_matrices=False)
print(f"奇异值占比: {(S**2 / np.sum(S**2)).round(4)}")
`

奇异值的平方占比等于 PCA 中的方差解释比。保留前 k 个奇异值就是保留最重要的 k 个"方向"。

### 推荐系统模拟

`python
R_filled = np.nan_to_num(R_missing, nan=0)
U_mat = svd_rec.fit_transform(R_filled)
R_reconstructed = U_mat @ V_mat
`

用户-物品评分矩阵有大量缺失值。SVD 降维后重建矩阵，缺失位置的重建值就是**预测评分**。这就是协同过滤推荐的基本原理。

### 文本 LSA（潜在语义分析）

`python
lsa = TruncatedSVD(n_components=2)
X_lsa = lsa.fit_transform(X_tfidf)
`

对 TF-IDF 矩阵做 SVD 降维，得到文档的"潜在语义"表示。语义相似的文档在降维空间中距离更近。

## 使用示例

`python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
X_reduced = svd.fit_transform(X_tfidf)  # 适合稀疏矩阵
`

## 注意事项

1. **TruncatedSVD 不做中心化**：与 PCA 不同，可直接用于稀疏矩阵
2. **PCA = 中心化 + SVD**：对稠密数据，PCA 和 TruncatedSVD 结果近似
3. **推荐系统**：实际中会加入正则化（SVD++、BiasSVD）
4. **计算复杂度**：完整 SVD 是 O(min(mn², m²n))，截断 SVD 更快

## 延伸思考

- **SVD++**：结合隐式反馈的推荐算法
- **NMF（非负矩阵分解）**：要求 U 和 V 非负，结果更可解释
- **随机化 SVD**：大数据集上的近似 SVD，sklearn.utils.extmath.randomized_svd
- **图像压缩**：保留前 k 个奇异值重建图像，实现有损压缩