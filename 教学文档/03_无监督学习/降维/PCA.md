# PCA 主成分分析：找到数据最有信息量的方向

> 所属模块：03_无监督学习/降维 | 源文件：PCA.py | 核心功能：方差解释比、白化、增量 PCA、作为预处理工具

## 概述

PCA（Principal Component Analysis）是最经典的降维算法。它寻找数据中**方差最大的正交方向**（主成分），将数据投影到这些方向上。方差最大意味着信息损失最小——因为方差代表数据的"分散程度"，分散越大，信息越多。

PCA 的应用场景远不止降维可视化：数据压缩、去噪、加速模型训练、去除多重共线性……它是数据科学家工具箱中的"瑞士军刀"。

脚本演示了 PCA 的完整工作流：从全部成分到选择性降维、白化处理、增量 PCA，以及作为 SVM 预处理的实际应用。

## 关键代码解释

### 方差解释比——选维度的依据

`python
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)
print(pca_full.explained_variance_ratio_)
`

每个主成分解释了多少原始方差。累积方差比达到 95% 通常是一个好的维度选择标准。

### 按阈值自动选维度

`python
pca_t = PCA(n_components=0.95)  # 保留 95% 方差
`

传入浮点数而非整数，PCA 会自动选择最少的维度使得累积方差比 >= 该阈值。

### 白化（Whitening）

`python
pca_white = PCA(n_components=2, whiten=True)
`

白化让降维后的各维度方差为 1 且互不相关。这在某些算法（如高斯混合模型）前做预处理很有用。

### 增量 PCA（IncrementalPCA）

`python
ipca = IncrementalPCA(n_components=2, batch_size=50)
`

标准 PCA 需要将全部数据加载到内存。IncrementalPCA 分批处理，适合大数据集。

## 使用示例

`python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
print(f"保留 {pca.explained_variance_ratio_.sum():.1%} 方差, 降到 {X_reduced.shape[1]} 维")
`

## 注意事项

1. **必须先特征缩放**：PCA 对尺度敏感，方差大的特征会主导主成分
2. **只能捕捉线性结构**：非线性流形结构用 t-SNE 或 UMAP
3. **主成分不一定是"有意义"的方向**：它们是数学上的最优方向，不一定有业务解释
4. **PCA 的 transform 必须用训练集的 PCA 对象**：避免数据泄露

## 延伸思考

- **核 PCA**：用核技巧处理非线性降维
- **稀疏 PCA**：让主成分只依赖少数特征，提高可解释性
- **PCA 与 SVD 的关系**：PCA = 中心化 + SVD
- **随机化 PCA**：svd_solver="randomized"，大数据集更快