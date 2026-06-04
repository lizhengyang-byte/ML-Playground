# KNN 分类：最近邻投票的懒惰学习者
> 难度标签：初级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：02_监督学习/分类 | 源文件：KNN_分类.py | 核心功能：K 值选择、距离度量、权重策略的全面对比

## 概述

KNN（K-Nearest Neighbors，K 近邻）是最直观的分类算法——"告诉我你的邻居是谁，我就知道你是谁"。给定一个待分类样本，找到训练集中距离最近的 K 个样本，让它们投票决定类别。

KNN 是**懒惰学习**（Lazy Learning）的代表：训练阶段什么都不做（只是存储数据），所有计算都推迟到预测时。这既是优点（实现简单、无需训练），也是缺点（预测慢、存储大）。

脚本在月亮形数据和 Iris 数据上对比了不同 K 值、距离度量和权重策略的影响。

## 代码结构

| 段落 | 内容 |
|------|------|
| 数据准备 | make_moons 月亮形数据 + Iris，均做 StandardScaler |
| K 值对比 | K 从 1 到 30 的训练/测试准确率变化 |
| 距离度量 | 曼哈顿（p=1）、欧氏（p=2）、闵可夫斯基（p=3） |
| 权重策略 | uniform（等权）vs distance（距离加权） |
| Iris 演示 | 完整的分类报告和概率预测 |

## 关键代码解释

### K 值的偏差-方差权衡

```python
for k in [1, 3, 5, 7, 10, 15, 20, 30]:
    knn = KNeighborsClassifier(n_neighbors=k)
```

- **K=1**：决策边界极度弯曲，完全记住训练集（过拟合）
- **K=N**：所有样本投票，永远预测多数类（欠拟合）
- **K 的选择**：通常用交叉验证在 [1, √n] 范围内搜索

### 距离加权

```python
knn_w = KNeighborsClassifier(n_neighbors=5, weights="distance")
```

uniform：K 个邻居每票等权。distance：权重 = 1/distance，越近的邻居话语权越大。这解决了 K 较大时远邻"稀释"近邻信号的问题。

### 特征缩放的必要性

KNN 基于距离计算，如果一个特征范围是 [0, 1000]，另一个是 [0, 1]，前者会完全主导距离。**StandardScaler 是 KNN 的前置必备步骤**。

## 使用示例

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
])
pipe.fit(X_train, y_train)
```

## 注意事项

1. **必须缩放**：距离计算受特征尺度直接影响
2. **维度灾难**：高维空间中距离变得无意义，KNN 在高维稀疏数据上表现很差
3. **预测速度**：朴素实现 O(n×d)，使用 KD-Tree 或 Ball-Tree 可加速到 O(d×log n)
4. **内存占用**：需要存储全部训练数据，大数据集不可行
5. **类别不平衡**：多数类天然占优势，考虑 weights="distance" 或对数据重采样

## 延伸思考

- **近似最近邻（ANN）**：FAISS、Annoy 等库用近似方法加速大规模最近邻搜索
- **KNN 回归**：将投票改为取均值/中位数，即可用于回归任务
- **距离度量学习**：学习一个自定义的距离函数，让同类样本更近、异类更远
- **加权 KNN 的概率校准**：distance 加权的 predict_proba 本身就是概率估计，通常比 uniform 更准确

## 数学原理

### 1. KNN 分类的投票机制

**代码对应**：`KNeighborsClassifier(n_neighbors=k, weights="uniform")` 和 `weights="distance"`。

对于查询样本 $\mathbf{x}$，找到 $K$ 个最近邻 $\mathcal{N}_K(\mathbf{x})$，分类规则为：

**多数投票**（`weights="uniform"`）：

$$\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_K(\mathbf{x})} \mathbb{I}(y_i = c)$$

**距离加权投票**（`weights="distance"`）：

$$\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_K(\mathbf{x})} \frac{1}{d(\mathbf{x}, \mathbf{x}_i)} \mathbb{I}(y_i = c)$$

**代码对应**：代码中对比了 `uniform` 和 `distance` 两种权重策略。距离加权让更近的邻居有更大话语权。

### 2. 距离度量

**代码对应**：`metric="minkowski", p=p` 定义了距离族。

闵可夫斯基距离：

$$d_p(\mathbf{x}, \mathbf{z}) = \left(\sum_{j=1}^{p} |x_j - z_j|^p\right)^{1/p}$$

- $p = 1$：曼哈顿距离（L1）
- $p = 2$：欧氏距离（L2，最常用）

余弦距离（文本分类常用）：

$$d_{\cos}(\mathbf{x}, \mathbf{z}) = 1 - \frac{\mathbf{x}^T\mathbf{z}}{\|\mathbf{x}\|_2 \|\mathbf{z}\|_2}$$

### 3. K 的偏差-方差权衡

**代码对应**：`for k in [1, 3, 5, ..., 30]` 展示了 K 对准确率的影响。

- $K = 1$：决策边界完全贴合训练数据（高方差，低偏差）
- $K$ 大：决策边界趋向线性/平滑（低方差，高偏差）
- 最优 $K$ 通过交叉验证选择

**贝叶斯误差率**：KNN 在 $N \to \infty$，$K \to \infty$，$K/N \to 0$ 时，误差率不超过贝叶斯最优分类器的 2 倍（Cover-Hart 定理）。

### 4. 维度灾难

高维空间中距离失去区分度。设数据均匀分布在 $[0,1]^p$，$K$ 个最近邻的平均距离为：

$$d_K \sim \left(\frac{K}{N}\right)^{1/p}$$

当 $p$ 增大时，最近邻和最远邻的距离趋于相同，KNN 的分类能力急剧下降。这也是 KNN 必须做特征缩放的原因——高维空间中各特征的尺度差异会进一步恶化距离计算。

### 5. 概率估计

**代码对应**：`knn.predict_proba(X_te[:3])` 返回各类别的概率估计。

KNN 的概率估计为 $K$ 个邻居中各类别的比例：

$$\hat{P}(y=c|\mathbf{x}) = \frac{|\{i \in \mathcal{N}_K(\mathbf{x}) : y_i = c\}|}{K}$$

或距离加权版本：

$$\hat{P}(y=c|\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_K(\mathbf{x})} w_i \cdot \mathbb{I}(y_i = c)}{\sum_{i \in \mathcal{N}_K(\mathbf{x})} w_i}$$

### 6. 训练与预测复杂度

| 阶段 | 朴素实现 | KD-Tree | Ball Tree |
|------|---------|---------|-----------|
| 训练 | $O(1)$ | $O(np\log n)$ | $O(np\log n)$ |
| 预测（每个样本） | $O(np)$ | $O(p\log n)$ | $O(p\log n)$ |

KNN 是**懒惰学习**：训练几乎无开销，但预测需要遍历（或搜索）所有训练样本。
