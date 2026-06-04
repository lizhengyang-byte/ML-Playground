# SVD 奇异值分解：矩阵的"基因组"
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：03_无监督学习/降维 | 源文件：SVD.py | 核心功能：完整 SVD 分解、截断 SVD、推荐系统模拟、文本 LSA

## 概述

SVD（Singular Value Decomposition）是线性代数中最重要的矩阵分解之一：X = U × Σ × Vᵀ。它把任意矩阵分解为三个矩阵的乘积——U 是左奇异向量，Σ 是奇异值（按降序排列），Vᵀ 是右奇异向量。

SVD 的应用极其广泛：PCA 的底层就是 SVD、推荐系统的核心是矩阵分解、图像压缩的本质是截断 SVD、文本处理的潜在语义分析（LSA）也是 SVD。

脚本演示了完整 SVD 分解、截断降维、推荐系统模拟和文本 LSA 四个应用场景。

## 关键代码解释

### 奇异值的能量分布

```python
U, S, Vt = svd(X_scaled, full_matrices=False)
print(f"奇异值占比: {(S**2 / np.sum(S**2)).round(4)}")
```

奇异值的平方占比等于 PCA 中的方差解释比。保留前 k 个奇异值就是保留最重要的 k 个"方向"。

### 推荐系统模拟

```python
R_filled = np.nan_to_num(R_missing, nan=0)
U_mat = svd_rec.fit_transform(R_filled)
R_reconstructed = U_mat @ V_mat
```

用户-物品评分矩阵有大量缺失值。SVD 降维后重建矩阵，缺失位置的重建值就是**预测评分**。这就是协同过滤推荐的基本原理。

### 文本 LSA（潜在语义分析）

```python
lsa = TruncatedSVD(n_components=2)
X_lsa = lsa.fit_transform(X_tfidf)
```

对 TF-IDF 矩阵做 SVD 降维，得到文档的"潜在语义"表示。语义相似的文档在降维空间中距离更近。

## 使用示例

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
X_reduced = svd.fit_transform(X_tfidf)  # 适合稀疏矩阵
```

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
## 数学原理

### 1. SVD 的数学定义

**代码对应**：sklearn 内部使用 SVD 实现 PCA。

任何 $n \times p$ 矩阵 $\mathbf{X}$ 都可以分解为：

$$\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^T$$

其中：
- $\mathbf{U} \in \mathbb{R}^{n \times r}$：左奇异向量（正交矩阵），$\mathbf{U}^T\mathbf{U} = \mathbf{I}$
- $\mathbf{D} \in \mathbb{R}^{r \times r}$：奇异值对角矩阵，$d_1 \geq d_2 \geq \cdots \geq d_r > 0$
- $\mathbf{V} \in \mathbb{R}^{p \times r}$：右奇异向量（正交矩阵），$\mathbf{V}^T\mathbf{V} = \mathbf{I}$
- $r = \text{rank}(\mathbf{X})$

### 2. 与特征值分解的关系

$$\mathbf{X}^T\mathbf{X} = \mathbf{V}\mathbf{D}^2\mathbf{V}^T, \quad \mathbf{X}\mathbf{X}^T = \mathbf{U}\mathbf{D}^2\mathbf{U}^T$$

即 $\mathbf{V}$ 是 $\mathbf{X}^T\mathbf{X}$ 的特征向量，$\mathbf{U}$ 是 $\mathbf{X}\mathbf{X}^T$ 的特征向量，$d_k^2$ 为对应的特征值。

**与 PCA 的联系**：PCA 的协方差矩阵 $\mathbf{\Sigma} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$，其特征向量即为 $\mathbf{V}$，特征值 $\lambda_k = d_k^2/(n-1)$。

### 3. 截断 SVD 与低秩近似

**Eckart-Young 定理**：在 Frobenius 范数下，秩 $k$ 的最优近似为截断 SVD：

$$\mathbf{X}_k = \sum_{i=1}^{k}d_i\mathbf{u}_i\mathbf{v}_i^T = \mathbf{U}_k\mathbf{D}_k\mathbf{V}_k^T$$

近似误差：

$$\|\mathbf{X} - \mathbf{X}_k\|_F = \sqrt{\sum_{i=k+1}^{r}d_i^2}$$

信息保留比：

$$\frac{\|\mathbf{X}_k\|_F^2}{\|\mathbf{X}\|_F^2} = \frac{\sum_{i=1}^{k}d_i^2}{\sum_{i=1}^{r}d_i^2}$$

### 4. SVD 在推荐系统中的应用

SVD 是协同过滤的核心算法。用户-物品评分矩阵 $\mathbf{R} \in \mathbb{R}^{m \times n}$ 的 SVD 近似：

$$\hat{R}_{ui} = \boldsymbol{\mu} + b_u + b_i + \mathbf{p}_u^T\mathbf{q}_i$$

其中 $\mathbf{p}_u$ 为用户 $u$ 的隐向量，$\mathbf{q}_i$ 为物品 $i$ 的隐向量。

### 5. 计算复杂度

完整 SVD：$O(\min(n^2p, np^2))$。截断 SVD（只求前 $k$ 个奇异值）：$O(npk)$。

sklearn 的 `TruncatedSVD` 使用随机化算法，适合大规模稀疏矩阵。
