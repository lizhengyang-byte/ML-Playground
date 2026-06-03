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