# 卷积神经网络 CNN：图像识别的王者
> 难度标签：高级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：06_深度学习 | 源文件：卷积神经网络_CNN.py | 核心功能：卷积层、池化层、特征提取、图像分类

## 概述

CNN 是处理网格数据（图像、音频）的专用神经网络。核心思想：用卷积核（小窗口）在输入上滑动，提取局部特征。通过多层卷积，从低级特征（边缘）逐步构建高级特征（物体部件）。

与 MLP 的关键区别：**局部连接**（只看小区域）+ **权值共享**（同一卷积核到处用）= 参数大幅减少。

## 关键代码解释

### 卷积层

```python
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
```

padding=1 保持空间尺寸不变。输出通道数 = 卷积核数量。

### 典型 CNN 架构

```python
Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Flatten -> Linear -> Output
```

## 注意事项

1. **输入格式**：(batch, channels, height, width)
2. **感受野**：深层卷积核看到的区域更大
3. **数据增强**：CNN 对平移、旋转敏感，需要数据增强

## 延伸思考

- **深度可分离卷积**：MobileNet 的核心，大幅减少参数
- **空洞卷积**：扩大感受野而不增加参数
- **注意力机制**：Vision Transformer 是否能取代 CNN
## 数学原理

### 1. 卷积操作

**代码对应**：`nn.Conv2d(in_channels, out_channels, kernel_size, padding)`。

2D 离散卷积：

$$(\mathbf{f} * \mathbf{g})(i, j) = \sum_m\sum_n f(m, n) \cdot g(i-m, j-n)$$

输出特征图：$O[c', i, j] = \sum_c\sum_m\sum_n W[c', c, m, n] \cdot I[c, i+m, j+n] + b[c']$

### 2. 输出尺寸计算

$$H_{\text{out}} = \left\lfloor\frac{H_{\text{in}} + 2 \cdot \text{padding} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor$$

**代码对应**：`padding=1, kernel_size=3, stride=1` 时输出尺寸不变。

### 3. 参数共享与局部连接

全连接层参数量：$H \times W \times C \times H' \times W' \times C'$

卷积层参数量：$K \times K \times C_{\text{in}} \times C_{\text{out}}$（与输入尺寸无关）

参数共享使 CNN 对平移具有等变性：如果输入平移，输出也相应平移。

### 4. 池化层

MaxPool：$O[i, j] = \max_{(m,n) \in R_{ij}} I[m, n]$

池化增加平移不变性，降低空间分辨率，减少计算量。

### 5. 感受野

第 $l$ 层神经元的感受野大小递推：

$$RF_l = RF_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1}s_i$$

深层神经元的感受野覆盖更大区域，能捕捉更全局的特征。

### 6. Dropout

训练时以概率 $p$ 随机将神经元输出置零：

$$h_i' = \begin{cases} 0 & \text{with prob } p \\ h_i / (1-p) & \text{with prob } 1-p \end{cases}$$

Dropup 是一种隐式集成（$2^n$ 个子网络的平均）。
