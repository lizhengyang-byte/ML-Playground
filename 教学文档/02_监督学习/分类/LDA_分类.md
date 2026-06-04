# LDA 分类：最大间距投影，一举两得的分类与降维
> 难度标签：初级 | 预计时长：15-25分钟 | 前置知识：无学习经验


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

```python
lda = LinearDiscriminantAnalysis(solver="svd")
lda.fit(X_train, y_train)
X_train_lda = lda.transform(X_train)
```

LDA 的投影维度上限是 min(n_features, n_classes - 1)。Iris 有 3 个类别，所以最多投影到 2 维。这意味着对于二分类问题，LDA 只能投影到 1 维。

### Shrinkage 收缩

```python
for shrinkage in [None, "auto", 0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    lda_sh = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
```

当样本量小而特征多时，协方差矩阵的估计不准确。Shrinkage 将协方差矩阵向对角矩阵收缩，等价于正则化。"auto" 使用 Ledoit-Wolf 公式自动选择收缩系数。

### LDA vs PCA 的关键差异

```python
pca = PCA(n_components=2)
# PCA：无监督，找方差最大方向
# LDA：有监督，找最能区分类别方向
```

PCA 最大化总方差，LDA 最大化类别可分性。如果类别间方差恰好在总方差最大方向上，两者结果相似。但当类别间的差异不是最大方差方向时（如类别差异在低方差子空间中），LDA 优势明显。

## 使用示例

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 作为分类器
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print(lda.predict(X_test))

# 作为降维器
X_reduced = lda.transform(X)  # 降到 n_classes - 1 维
```

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

## 数学原理

### 1. LDA 的目标：最大化 Fisher 判别准则

**代码对应**：`LinearDiscriminantAnalysis(solver="svd")` 训练 LDA。

LDA 寻找投影方向 $\mathbf{w}$，使投影后**类间散度**最大、**类内散度**最小：

$$J(\mathbf{w}) = \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$$

其中：

**类内散度矩阵**（Within-class scatter）：

$$\mathbf{S}_W = \sum_{k=1}^{K}\sum_{\mathbf{x}_i \in C_k}(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$$

**类间散度矩阵**（Between-class scatter）：

$$\mathbf{S}_B = \sum_{k=1}^{K} n_k(\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$$

其中 $\boldsymbol{\mu}_k$ 为类别 $k$ 的均值，$\boldsymbol{\mu}$ 为全局均值。

### 2. 求解：广义特征值问题

最大化 $J(\mathbf{w})$ 等价于求解广义特征值问题：

$$\mathbf{S}_B\mathbf{w} = \lambda\mathbf{S}_W\mathbf{w}$$

当 $\mathbf{S}_W$ 可逆时，等价于标准特征值问题：

$$\mathbf{S}_W^{-1}\mathbf{S}_B\mathbf{w} = \lambda\mathbf{w}$$

选取前 $d$ 个最大特征值对应的特征向量作为投影方向，其中 $d \leq K - 1$。

**代码对应**：Iris 3 类最多投影到 $3 - 1 = 2$ 维。`lda.explained_variance_ratio_` 返回各方向的方差解释比。

### 3. LDA 作为分类器

LDA 假设各类别服从同协方差的高斯分布：

$$P(\mathbf{x}|y=k) = \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma})$$

后验概率为：

$$\ln P(y=k|\mathbf{x}) = \ln\pi_k + \boldsymbol{\mu}_k^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - \frac{1}{2}\boldsymbol{\mu}_k^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k + \text{const}$$

这是 $\mathbf{x}$ 的**线性函数**，因此决策边界是超平面——这就是"线性判别分析"名称的由来。

### 4. Shrinkage（收缩正则化）

**代码对应**：`shrinkage="auto"` 或具体数值。

当样本量小而特征数多时，$\mathbf{S}_W$ 估计不准（甚至不可逆）。Shrinkage 将 $\mathbf{S}_W$ 向对角矩阵收缩：

$$\mathbf{S}_W^{\text{shrunk}} = (1 - \alpha)\mathbf{S}_W + \alpha\frac{\text{tr}(\mathbf{S}_W)}{p}\mathbf{I}$$

其中 $\alpha \in [0, 1]$ 为收缩系数。$\alpha = 0$ 无收缩，$\alpha = 1$ 完全对角（假设特征独立）。

`shrinkage="auto"` 使用 Ledoit-Wolf 引理自动估计最优 $\alpha$。

### 5. LDA vs PCA

| 特性 | PCA | LDA |
|------|-----|-----|
| 目标 | 最大化投影方差 | 最大化类间/类内散度比 |
| 监督 | 无监督 | 有监督 |
| 最大维度 | $p$（特征数） | $K-1$（类别数 - 1） |
| 适用场景 | 降维、去相关 | 降维 + 分类 |

**代码对应**：代码中对比了 PCA 和 LDA 降维后 KNN 分类的效果——LDA 降维后的分类准确率通常更高，因为投影方向考虑了类别信息。

### 6. QDA（二次判别分析）

如果各类别的协方差矩阵**不**相同（$\boldsymbol{\Sigma}_k$ 各异），则决策边界变为**二次曲面**，这就是 QDA。sklearn 提供 `QuadraticDiscriminantAnalysis`。

QDA 的判别函数为：

$$\delta_k(\mathbf{x}) = -\frac{1}{2}\ln|\boldsymbol{\Sigma}_k| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T\boldsymbol{\Sigma}_k^{-1}(\mathbf{x} - \boldsymbol{\mu}_k) + \ln\pi_k$$

LDA 假设 $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$（共同协方差），因此 $\ln|\boldsymbol{\Sigma}_k|$ 和 $\boldsymbol{\Sigma}_k^{-1}$ 项在各类别间抵消，判别函数退化为线性。
