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
﻿## 数学原理

### 1. 前向传播

**代码对应**：MLP 由多层全连接层组成。

单层前向传播：$\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$

多层（$L$ 层）：
$$\mathbf{h}^{(0)} = \mathbf{x}, \quad \mathbf{h}^{(l)} = \sigma(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}), \quad \hat{y} = \mathbf{h}^{(L)}$$

常用激活函数：
- ReLU：$\sigma(z) = \max(0, z)$
- Sigmoid：$\sigma(z) = 1/(1+e^{-z})$
- Tanh：$\sigma(z) = (e^z - e^{-z})/(e^z + e^{-z})$

### 2. 反向传播（Backpropagation）

**链式法则**：$\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \frac{\partial L}{\partial \mathbf{h}^{(l)}} \cdot \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{W}^{(l)}}$

误差从输出层逐层向前传播：
$$\boldsymbol{\delta}^{(L)} = \nabla_{\hat{y}}L \odot \sigma'(\mathbf{z}^{(L)})$$
$$\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)T}\boldsymbol{\delta}^{(l+1)}) \odot \sigma'(\mathbf{z}^{(l)})$$

### 3. 梯度下降优化

**SGD**：$\theta \leftarrow \theta - \alpha\nabla_\theta L$

**Adam**：结合动量和自适应学习率：
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t)$$
$$\theta \leftarrow \theta - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 4. 梯度消失与梯度爆炸

梯度在反向传播中逐层相乘：$\frac{\partial L}{\partial \mathbf{W}^{(1)}} \propto \prod_{l=1}^{L-1}\mathbf{W}^{(l)}\text{diag}(\sigma'(\mathbf{z}^{(l)}))$

- Sigmoid/Tanh 的导数 $\leq 0.25$，连乘后梯度指数衰减（梯度消失）
- ReLU 的导数为 0 或 1，有效缓解梯度消失
- Batch Normalization 也帮助稳定梯度
