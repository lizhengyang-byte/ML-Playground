# 卷积神经网络 CNN：图像识别的王者

> 所属模块：06_深度学习 | 源文件：卷积神经网络_CNN.py | 核心功能：卷积层、池化层、特征提取、图像分类

## 概述

CNN 是处理网格数据（图像、音频）的专用神经网络。核心思想：用卷积核（小窗口）在输入上滑动，提取局部特征。通过多层卷积，从低级特征（边缘）逐步构建高级特征（物体部件）。

与 MLP 的关键区别：**局部连接**（只看小区域）+ **权值共享**（同一卷积核到处用）= 参数大幅减少。

## 关键代码解释

### 卷积层

`python
nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
`

padding=1 保持空间尺寸不变。输出通道数 = 卷积核数量。

### 典型 CNN 架构

`python
Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Flatten -> Linear -> Output
`

## 注意事项

1. **输入格式**：(batch, channels, height, width)
2. **感受野**：深层卷积核看到的区域更大
3. **数据增强**：CNN 对平移、旋转敏感，需要数据增强

## 延伸思考

- **深度可分离卷积**：MobileNet 的核心，大幅减少参数
- **空洞卷积**：扩大感受野而不增加参数
- **注意力机制**：Vision Transformer 是否能取代 CNN