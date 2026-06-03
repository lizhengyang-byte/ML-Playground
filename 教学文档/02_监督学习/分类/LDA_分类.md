# LDA 分类：最大间距投影，一举两得的分类与降维

> 所属模块：02_监督学习/分类 | 源文件：LDA_分类.py | 核心功能：LDA 分类、有监督降维、与 PCA 对比

## 概述

线性判别分析（Linear Discriminant Analysis, LDA）是一个"两栖"算法：它既可以作为**分类器**，也可以作为**有监督降维**工具。

核心思想：找到一个投影方向，使得投影后**类间方差最大、类内方差最小**。用数学语言说，就是最大化 Fisher 判别准则：J(w) = (类间散度) / (类内散度)。

与 PCA 的区别很关键：PCA 找的是**方差最大**的方向（无监督），LDA 找的是**最能区分不同类别**的方向（有监督）。在分类任务中，LDA 投影通常比 PCA 投影更有判别力。

## 代码结构

| 段落 | 内容 |
|------|------|
| LDA 分类 | Iris 数据集上的训练和评估 |
| LDA 降维 | 4 维 → 2 维（类别数 - 1） |
| LDA vs PCA 对比 | 降维后接 KNN 分类的准确率对比 |
| 求解器对比 | svd、lsqr、eigen |
| Shrinkage 参数 | 正则化收缩系数对小样本的影响 |

## 关键代码解释

### 降维上限 = 类别数 - 1

`python
lda = LinearDiscriminantAnalysis(solver="svd")
lda.fit(X_train, y_train)
X_train_lda = lda.transform(X_train)
`

LDA 的投影维度上限是 min(n_features, n_classes - 1)。Iris 有 3 个类别，所以最多投影到 2 维。这意味着对于二分类问题，LDA 只能投影到 1 维。

### Shrinkage 收缩

`python
for shrinkage in [None, "auto", 0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    lda_sh = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
`

当样本量小而特征多时，协方差矩阵的估计不准确。Shrinkage 将协方差矩阵向对角矩阵收缩，等价于正则化。"auto" 使用 Ledoit-Wolf 公式自动选择收缩系数。

### LDA vs PCA 的关键差异

`python
pca = PCA(n_components=2)
# PCA：无监督，找方差最大方向
# LDA：有监督，找最能区分类别方向
`

PCA 最大化总方差，LDA 最大化类别可分性。如果类别间方差恰好在总方差最大方向上，两者结果相似。但当类别间的差异不是最大方差方向时（如类别差异在低方差子空间中），LDA 优势明显。

## 使用示例

`python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 作为分类器
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print(lda.predict(X_test))

# 作为降维器
X_reduced = lda.transform(X)  # 降到 n_classes - 1 维
`

## 注意事项

1. **正态假设**：LDA 假设各类别数据服从同协方差的高斯分布，违反时效果下降
2. **降维上限**：最多 n_classes - 1 维，二分类只能降到 1 维
3. **Shrinkage 只用于 lsqr 和 eigen 求解器**，svd 不支持
4. **对异常值敏感**：协方差矩阵的估计受异常值影响
5. **样本量要求**：每个类别的样本数应大于特征数，否则协方差矩阵不可逆

## 延伸思考

- **QDA（二次判别分析）**：放松等协方差假设，允许每个类别有自己的协方差矩阵，决策边界为二次曲面
- **Fisher 判别分析 vs LDA**：Fisher LDA 只做投影不做概率建模，概率 LDA 加入了贝叶斯推断
- **Kernel LDA**：用核技巧处理非线性判别问题
- **LDA 在人脸识别中的应用**："Fisherfaces" 就是 LDA 降维 + 最近邻分类的经典方法