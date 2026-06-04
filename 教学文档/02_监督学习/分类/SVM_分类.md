# SVM 分类：最大间隔的优雅分类器
> 难度标签：初级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：02_监督学习/分类 | 源文件：SVM_分类.py | 核心功能：4 种核函数对比、C/gamma 调参、多分类支持

## 概述

支持向量机（Support Vector Machine, SVM）是机器学习中**最优雅**的算法之一。核心思想：找到一个超平面，使得两类数据之间的**间隔（margin）最大化**。那些恰好落在间隔边界上的训练样本，就叫做"支持向量"——它们是决定分类边界的关键样本。

但 SVM 真正的杀手锏是**核技巧（Kernel Trick）**：通过核函数将数据映射到高维空间，使原本线性不可分的数据变得线性可分——而且不需要真的计算高维坐标，只需计算核函数值。

脚本在三种不同形状的数据（线性可分、月亮形、同心圆）上对比了 4 种核函数，并深入分析了 C 和 gamma 参数的影响。

## 代码结构

| 段落 | 内容 |
|------|------|
| 数据准备 | 线性可分、月亮形（make_moons）、同心圆（make_circles） |
| 核函数对比 | linear、poly、rbf、sigmoid 在三种数据上的表现 |
| C 参数对比 | 从 0.01 到 100，观察间隔大小和支持向量数变化 |
| Iris 多分类 | RBF 核 + One-vs-One 策略 |
| gamma 参数 | 从 scale 到 10.0，观察决策边界的弯曲程度 |

## 关键代码解释

### 核函数选择

```python
kernels = ["linear", "poly", "rbf", "sigmoid"]
```

- **linear**：线性核，适合高维数据（特征数 > 样本数）或线性可分数据
- **poly**：多项式核，捕获多项式交互特征
- **rbf**：高斯核（默认），万能核，适合大多数非线性问题
- **sigmoid**：类似神经网络的激活函数，较少使用

### C 参数——间隔与误分类的平衡

```python
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    svm_c = SVC(kernel="linear", C=C, random_state=42)
```

C 越大，对误分类的惩罚越重，间隔越窄（硬间隔），容易过拟合。C 越小，允许更多误分类，间隔越宽（软间隔），泛化更好。

### gamma 参数——RBF 核的"视野"

```python
for gamma in ["scale", "auto", 0.01, 0.1, 1.0, 10.0]:
    svm_g = SVC(kernel="rbf", C=10.0, gamma=gamma, random_state=42)
```

gamma 控制单个训练样本的"影响范围"。gamma 越大，影响范围越小，决策边界越弯曲（过拟合）。gamma 越小，影响范围越大，边界越平滑（欠拟合）。

## 使用示例

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True))
])
pipe.fit(X_train, y_train)
print(pipe.predict_proba(X_test))
```

## 注意事项

1. **必须特征缩放**：SVM 基于距离计算，对特征尺度极其敏感
2. **大数据集慎用**：训练复杂度 O(n² ~ n³)，万级以上样本考虑 SGDClassifier 或 LinearSVC
3. **概率输出需额外计算**：probability=True 使用 Platt Scaling，有额外开销且不一定准确
4. **C 和 gamma 配合调参**：通常用网格搜索 + 交叉验证同时优化两个参数
5. **多分类策略**：SVM 默认使用 One-vs-One（每对类别训练一个分类器），多分类时分类器数量为 n_classes × (n_classes-1) / 2

## 延伸思考

- **LinearSVC vs SVC(kernel="linear")**：LinearSVC 使用 liblinear，训练更快，适合大规模线性问题
- **核近似**：Nystroem 和 RBFSampler 可以近似核映射，将 SVM 与线性模型结合，加速大规模训练
- **SVM 的几何直觉**：间隔最大化等价于在约束下最小化权重范数 ||w||²，这是一个凸优化问题
- **SVM 与神经网络的关系**：使用 RBF 核的 SVM 可以看作一个单隐层 RBF 网络

## 数学原理

### 1. 硬间隔 SVM 的优化问题

**代码对应**：`SVC(kernel="linear", C=1.0)` 中 $C \to \infty$ 时退化为硬间隔。

对于线性可分数据，SVM 寻找使分类间隔最大化的超平面 $\mathbf{w}^T\mathbf{x} + b = 0$：

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2$$

$$\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n$$

**间隔的几何意义**：正类支持向量到超平面的距离为 $1/\|\mathbf{w}\|$，负类也是 $1/\|\mathbf{w}\|$，总间隔为 $2/\|\mathbf{w}\|$。最大化间隔等价于最小化 $\|\mathbf{w}\|^2$。

### 2. 软间隔 SVM

**代码对应**：`C` 参数控制软间隔的松弛程度。

引入松弛变量 $\xi_i \geq 0$ 允许误分类：

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|_2^2 + C\sum_{i=1}^{n}\xi_i$$

$$\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

- $C$ 大 $\Rightarrow$ 对误分类惩罚重，间隔窄（硬间隔倾向）
- $C$ 小 $\Rightarrow$ 允许更多误分类，间隔宽（软间隔）

**代码对应**：代码中 `for C in [0.01, 0.1, 1.0, 10.0, 100.0]` 展示了 C 对间隔和支持向量数的影响。

### 3. 对偶问题

通过拉格朗日乘子法，原始问题的对偶形式为：

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$$

$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^{n}\alpha_i y_i = 0$$

**KKT 条件**：$\alpha_i > 0$ 的样本即为**支持向量**——它们恰好落在间隔边界上或被误分类。

决策函数：

$$f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV}\alpha_i y_i \mathbf{x}_i^T\mathbf{x} + b\right)$$

### 4. 核技巧

**代码对应**：`kernel` 参数（linear、poly、rbf、sigmoid）。

核技巧将 $\mathbf{x}_i^T\mathbf{x}_j$ 替换为核函数 $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$：

$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

常用核函数：

| 核函数 | 公式 | 参数 |
|--------|------|------|
| 线性核 | $K(\mathbf{x}, \mathbf{z}) = \mathbf{x}^T\mathbf{z}$ | 无 |
| 多项式核 | $K(\mathbf{x}, \mathbf{z}) = (\gamma\mathbf{x}^T\mathbf{z} + r)^d$ | $d, r$ |
| RBF 核 | $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma\|\mathbf{x}-\mathbf{z}\|^2)$ | $\gamma$ |
| Sigmoid 核 | $K(\mathbf{x}, \mathbf{z}) = \tanh(\gamma\mathbf{x}^T\mathbf{z} + r)$ | $\gamma, r$ |

### 5. gamma 参数的几何意义

**代码对应**：`for gamma in ["scale", "auto", 0.01, ...]` 展示了 gamma 的影响。

RBF 核中 $\gamma = 1/(2\sigma^2)$，控制单个样本的"影响半径"：

$$K(\mathbf{x}, \mathbf{x}_i) = \exp\left(-\frac{\|\mathbf{x}-\mathbf{x}_i\|^2}{2\sigma^2}\right)$$

- $\gamma$ 大（$\sigma$ 小）：核函数窄，每个样本只影响附近，决策边界弯曲（过拟合）
- $\gamma$ 小（$\sigma$ 大）：核函数宽，影响范围大，决策边界平滑（欠拟合）

### 6. 训练复杂度与多分类策略

训练复杂度约 $O(n^2 p)$ 到 $O(n^3)$。sklearn 默认使用 **One-vs-One**（OvO）多分类策略：$K$ 个类别需要训练 $K(K-1)/2$ 个二分类器。

**代码对应**：Iris 3 类需要 $3 \times 2 / 2 = 3$ 个分类器。
