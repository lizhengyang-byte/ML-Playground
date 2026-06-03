# ResNet：跳跃连接让网络可以无限深

> 所属模块：06_深度学习 | 源文件：残差网络_ResNet.py | 核心功能：残差块、跳跃连接、梯度流通

## 概述

ResNet（2015）通过**跳跃连接**（skip connection）解决了深层网络的退化问题：不是直接学习映射 H(x)，而是学习残差 F(x) = H(x) - x，然后 H(x) = F(x) + x。

这让梯度可以通过跳跃连接直接流回前面的层，解决了深层网络的梯度消失问题。ResNet-152 在 ImageNet 上达到了 3.57% 的错误率。

## 关键代码解释

### 残差块

`python
class ResBlock(nn.Module):
    def forward(self, x):
        return F.relu(self.conv2(self.conv1(x)) + x)  # 跳跃连接
`

输入 x 直接加到卷积输出上——如果卷积学到的是 0（恒等映射），输出就等于输入，不会退化。

## 注意事项

1. **Batch Normalization 是标配**：每个卷积后加 BN
2. **深度不是越深越好**：ResNet-1000 可能不如 ResNet-101
3. **Pre-activation ResNet**：BN+ReLU 放在卷积前，效果有时更好

## 延伸思考

- **DenseNet**：所有层互相连接
- **ResNeXt**：分组卷积 + 残差
- **EfficientNet**：系统化地缩放网络宽度、深度和分辨率
﻿## 数学原理

### 1. 深层网络的退化问题

理论上更深的网络不应比浅网络差（多出的层可以学习恒等映射），但实际训练中：
- 深层网络的训练误差反而更高（退化问题）
- 这不是过拟合，而是**优化困难**

ResNet 的核心假设：让网络学习**残差**比学习完整映射更容易。

### 2. 残差学习框架

设期望学习的底层映射为 $H(x)$，ResNet 将其重参数化为：

$$H(x) = F(x) + x$$

其中 $F(x) = H(x) - x$ 是**残差函数**。网络只需学习残差 $F(x)$，而非完整映射 $H(x)$。

当最优映射接近恒等时，$F(x) \to 0$ 比 $H(x) \to x$ 更容易通过权重参数化实现。

### 3. 残差块的前向传播

代码中基础残差块的数学表达：

$$y = \mathcal{F}(x, \{W_i\}) + x$$

其中：

$$\mathcal{F}(x, \{W_1, W_2\}) = W_2 \sigma(BN(W_1 x))$$

展开为：
1. $h_1 = \text{BN}(W_1 * x)$ （卷积 + 批归一化）
2. $a_1 = \text{ReLU}(h_1)$ （激活）
3. $h_2 = \text{BN}(W_2 * a_1)$ （第二层卷积 + BN）
4. $y = \text{ReLU}(h_2 + x)$ （跳跃连接 + 激活）

代码对应：
```python
def forward(self, x):
    return self.relu(self.block(x) + x)  # F(x) + x
```

### 4. 梯度传播分析

残差连接对梯度流的影响。设损失为 $\mathcal{L}$，对 $x$ 求梯度：

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial x}\right)$$

关键：梯度中有一个**恒为 1 的直接路径**，即使 $\frac{\partial \mathcal{F}}{\partial x} \approx 0$（梯度消失），信号仍能通过跳跃连接传播。

### 5. Batch Normalization 的数学

代码中每层卷积后都有 BN：

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

其中 $\mu_\mathcal{B}, \sigma_\mathcal{B}^2$ 是 mini-batch 的均值和方差，$\gamma, \beta$ 是可学习参数。

BN 的作用：
- 减少内部协变量偏移（internal covariate shift）
- 允许使用更大的学习率
- 隐式正则化效果

### 6. 全局平均池化

代码中 `AdaptiveAvgPool2d(1)` 将特征图压缩为标量：

$$z_c = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W} x_{c,i,j}$$

对每个通道 $c$ 取空间维度的平均值，参数量为零，减少过拟合。

### 7. 残差块的维度匹配

当输入输出维度相同时（代码中的情况），跳跃连接直接相加：

$$y = \mathcal{F}(x) + x, \quad \dim(\mathcal{F}(x)) = \dim(x)$$

当维度不同时（如通道数变化），需要投影 shortcut：

$$y = \mathcal{F}(x) + W_s x$$

其中 $W_s$ 是 $1 \times 1$ 卷积用于调整维度。

### 8. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `Conv2d(channels, channels, 3, padding=1)` | $W * x$，$3 \times 3$ 卷积保持空间尺寸 |
| `BatchNorm2d(channels)` | 批归一化 $\hat{x} = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$，可学习 $\gamma, \beta$ |
| `self.block(x) + x` | 残差连接 $\mathcal{F}(x) + x$ |
| `AdaptiveAvgPool2d(1)` | 全局平均池化 $z_c = \frac{1}{HW}\sum x_{c,i,j}$ |
| `bias=False` | BN 层自带偏置，卷积无需 $b$ |
| `ResidualBlock(32)` × 2 | 堆叠两个残差块，加深网络 |
| `PlainNet` 对比 | 无 $+x$ 跳跃连接的普通网络，验证残差优势 |
