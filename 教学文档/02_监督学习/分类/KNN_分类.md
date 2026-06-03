# KNN 分类：最近邻投票的懒惰学习者

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

`python
for k in [1, 3, 5, 7, 10, 15, 20, 30]:
    knn = KNeighborsClassifier(n_neighbors=k)
`

- **K=1**：决策边界极度弯曲，完全记住训练集（过拟合）
- **K=N**：所有样本投票，永远预测多数类（欠拟合）
- **K 的选择**：通常用交叉验证在 [1, √n] 范围内搜索

### 距离加权

`python
knn_w = KNeighborsClassifier(n_neighbors=5, weights="distance")
`

uniform：K 个邻居每票等权。distance：权重 = 1/distance，越近的邻居话语权越大。这解决了 K 较大时远邻"稀释"近邻信号的问题。

### 特征缩放的必要性

KNN 基于距离计算，如果一个特征范围是 [0, 1000]，另一个是 [0, 1]，前者会完全主导距离。**StandardScaler 是 KNN 的前置必备步骤**。

## 使用示例

`python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance"))
])
pipe.fit(X_train, y_train)
`

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