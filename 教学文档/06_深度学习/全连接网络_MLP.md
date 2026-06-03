# 全连接网络 MLP：深度学习的第一步

> 所属模块：06_深度学习 | 源文件：全连接网络_MLP.py | 核心功能：PyTorch 全连接层、前向传播、训练循环

## 概述

多层感知机（Multi-Layer Perceptron, MLP）是最基础的深度学习模型。它由多个全连接层组成，每层的每个神经元与上一层所有神经元相连。虽然简单，但 MLP 是理解深度学习的起点——CNN、RNN、Transformer 都是在此基础上的改进。

## 关键代码解释

### 典型 MLP 结构

`python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
`

每层结构：线性变换 + 非线性激活。没有激活函数，多层线性变换等价于单层。

### 训练循环

`python
for epoch in range(epochs):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
`

## 注意事项

1. **激活函数必须有**：否则多层等价于单层
2. **梯度消失**：Sigmoid/Tanh 在深层网络中梯度消失，用 ReLU
3. **权重初始化**：不当初始化会导致训练失败

## 延伸思考

- **通用近似定理**：单隐层 MLP 可以逼近任意连续函数
- **Batch Normalization**：加速训练、稳定梯度
- **Dropout**：正则化手段，随机丢弃神经元