# 高斯混合模型 GMM：概率版的 KMeans
> 难度标签：中级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：03_无监督学习/聚类 | 源文件：高斯混合模型_GMM.py | 核心功能：软聚类、BIC/AIC 选模型、协方差类型对比、生成新样本

## 概述

如果 KMeans 是"硬聚类"（每个点属于且只属于一个簇），GMM 就是"软聚类"——每个点以不同概率属于各个簇。GMM 假设数据由 K 个高斯分布混合生成，用 EM（Expectation-Maximization）算法估计每个高斯的均值、协方差和混合权重。

与 KMeans 的关键区别：
- KMeans 只能发现球形簇，GMM 可以发现椭圆形簇（由协方差矩阵控制形状）
- KMeans 给出硬标签，GMM 给出概率分布
- GMM 可以生成新样本

脚本演示了四种协方差类型、BIC/AIC 选最优 K、软聚类概率输出和从模型生成新样本。

## 关键代码解释

### BIC/AIC 选最优 K

```python
for k in range(1, 10):
    gmm_k = GaussianMixture(n_components=k, random_state=42, n_init=10)
    bic_scores.append(gmm_k.bic(X))
```

BIC（贝叶斯信息准则）和 AIC（赤池信息准则）都在对数似然的基础上惩罚模型复杂度。BIC 惩罚更重，倾向选择更简单的模型。**BIC 最小的 K 通常是好选择**。

### 四种协方差类型

```python
for cov_type in ["full", "tied", "diag", "spherical"]:
```

- **full**：每个簇有自己的完整协方差矩阵（最灵活，参数最多）
- **tied**：所有簇共享同一个协方差矩阵
- **diag**：对角协方差（特征独立）
- **spherical**：各方向等方差（最简单，接近 KMeans）

### 软聚类

```python
probs = gmm_final.predict_proba(X)
```

每个样本得到一个概率向量，表示它属于各个簇的概率。如果某个样本的 [0.45, 0.40, 0.10, 0.05] 意味着它在簇 0 和簇 1 之间"犹豫不决"——这比 KMeans 的硬标签提供了更多信息。

## 使用示例

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4, covariance_type="full", n_init=10)
labels = gmm.fit_predict(X)
probs = gmm.predict_proba(X)
```

## 注意事项

1. **对初始值敏感**：
_init=10 运行多次取最优
2. **BIC 比 AIC 更保守**：倾向于选更少的簇
3. **full 协方差在高维时参数爆炸**：d 维有 O(d²) 个参数
4. **EM 可能收敛到局部最优**
5. **可以生成新样本**：gmm.sample(n)

## 延伸思考

- **变分高斯混合模型（VBGMM）**：贝叶斯版本，自动确定有效成分数
- **GMM 用于异常检测**：低似然度的点可能是异常值
- **GMM 用于语音识别**：说话人识别中的经典方法
- **贝叶斯高斯混合**：BayesianGaussianMixture，自动确定有效簇数
## 数学原理

### 1. 高斯混合模型的概率框架

**代码对应**：`GaussianMixture(n_components=k)` 训练 GMM。

GMM 假设数据由 $K$ 个高斯分布混合生成：

$$P(\mathbf{x}) = \sum_{k=1}^{K}\pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

其中 $\pi_k$ 为混合权重（$\sum_k \pi_k = 1$），$\boldsymbol{\mu}_k$ 和 $\boldsymbol{\Sigma}_k$ 为第 $k$ 个高斯分量的均值和协方差。

### 2. EM 算法求解

**E 步**（Expectation）：计算样本 $i$ 属于分量 $k$ 的**后验概率**（责任度）：

$$\gamma_{ik} = P(z_i = k|\mathbf{x}_i) = \frac{\pi_k \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K}\pi_j \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M 步**（Maximization）：用责任度更新参数：

$$\boldsymbol{\mu}_k^{\text{new}} = \frac{\sum_i \gamma_{ik}\mathbf{x}_i}{\sum_i \gamma_{ik}}, \quad \boldsymbol{\Sigma}_k^{\text{new}} = \frac{\sum_i \gamma_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^T}{\sum_i \gamma_{ik}}$$

$$\pi_k^{\text{new}} = \frac{\sum_i \gamma_{ik}}{n}$$

EM 算法保证对数似然单调递增，但只能收敛到局部最优。

### 3. 对数似然

$$\ell(\boldsymbol{\theta}) = \sum_{i=1}^{n}\ln\left[\sum_{k=1}^{K}\pi_k \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)\right]$$

### 4. 协方差类型

**代码对应**：`covariance_type` 参数（full、tied、diag、spherical）。

| 类型 | $\boldsymbol{\Sigma}_k$ 形状 | 参数数 | 适用场景 |
|------|---------------------------|--------|---------|
| full | 每个分量独立完整协方差 | $Kp(p+1)/2$ | 一般情况 |
| tied | 所有分量共享协方差 | $p(p+1)/2$ | 各簇形状相似 |
| diag | 对角协方差 | $Kp$ | 特征独立 |
| spherical | $\sigma_k^2\mathbf{I}$ | $K$ | 球形簇（类似 KMeans） |

### 5. 模型选择：BIC/AIC

**代码对应**：`gmm.bic(X)` 和 `gmm.aic(X)` 用于选择最优 $K$。

$$\text{BIC} = -2\ell(\hat{\boldsymbol{\theta}}) + d\ln n$$

$$\text{AIC} = -2\ell(\hat{\boldsymbol{\theta}}) + 2d$$

其中 $d$ 为参数总数。BIC 和 AIC 越小越好，BIC 对复杂模型惩罚更重。

### 6. GMM vs KMeans

KMeans 是 GMM 的特殊情况：当所有 $\boldsymbol{\Sigma}_k = \sigma^2\mathbf{I}$ 且 $\sigma \to 0$ 时，GMM 退化为 KMeans（硬分配）。

GMM 的优势：
- **软分配**：给出样本属于各簇的概率（不仅是标签）
- **椭圆形簇**：通过协方差矩阵捕捉非球形结构
- **概率模型**：可用 BIC/AIC 选择簇数
