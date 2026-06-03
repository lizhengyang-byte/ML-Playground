# 变分自编码器 VAE：学习生成数据

> 所属模块：06_深度学习 | 源文件：变分自编码器_VAE.py | 核心功能：概率编码、重参数化技巧、ELBO 损失、样本生成

## 概述

VAE 在自编码器基础上引入概率建模。编码器不输出确定性向量，而是输出均值和方差，定义一个高斯分布。从该分布采样得到潜在向量，解码器重建。可以生成全新样本。

## 关键代码解释

### 重参数化技巧

`python
z = mu + sigma * epsilon  # epsilon ~ N(0,1)
`

直接从高斯分布采样不可微。重参数化技巧将随机性转移到 epsilon，使梯度可以反向传播。

### ELBO 损失

`python
loss = reconstruction_loss + KL_divergence
`

重建损失让输出接近输入，KL 散度让潜在分布接近标准正态。

## 注意事项

1. **KL 散度的权重**：太大会导致"后验坍塌"（忽略输入）
2. **生成质量不如 GAN**：但训练更稳定
3. **潜在空间有结构**：可以插值生成过渡样本

## 延伸思考

- **条件 VAE**：给定标签生成特定类别
- **VQ-VAE**：离散潜在空间，Stable Diffusion 的基础
- **扩散模型**：比 VAE 更强大的生成模型
﻿## 数学原理

### 1. VAE 的概率图模型

VAE 假设数据 $x$ 由潜变量 $z$ 生成，其联合分布为：

$$p_\theta(x, z) = p_\theta(x|z) p(z)$$

其中先验 $p(z) = \mathcal{N}(0, I)$。目标是最大化边际对数似然：

$$\log p_\theta(x) = \log \int p_\theta(x|z) p(z) dz$$

直接计算积分不可行（需要遍历所有 $z$），因此引入变分推断。

### 2. 证据下界（ELBO）

用近似后验 $q_\phi(z|x)$（编码器）替代真实后验 $p_\theta(z|x)$，推导出对数似然的下界：

$$\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建项}} - \underbrace{KL(q_\phi(z|x) \| p(z))}_{\text{正则项}}$$

**推导过程**（KL 散度分解）：

$$KL(q_\phi(z|x) \| p_\theta(z|x)) = \mathbb{E}_{q_\phi}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right] = \mathbb{E}_{q_\phi}\left[\log \frac{q_\phi(z|x) p(x)}{p_\theta(x,z)}\right]$$

整理得：

$$\log p(x) = \mathcal{L}(\theta, \phi; x) + KL(q_\phi(z|x) \| p_\theta(z|x))$$

由于 KL 散度 $\geq 0$，$\mathcal{L}$ 是 $\log p(x)$ 的下界。

### 3. 重参数化技巧

ELBO 中的期望 $\mathbb{E}_{q_\phi(z|x)}$ 需要对 $z$ 采样，但采样操作不可微。**重参数化技巧**：

$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

将随机性转移到 $\epsilon$，使 $z$ 对 $\mu, \sigma$ 可微。代码中：

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)    # σ = exp(½ log σ²)
    eps = torch.randn_like(std)       # ε ~ N(0, I)
    return mu + eps * std             # z = μ + σ·ε
```

### 4. 两项损失的具体形式

**KL 散度项**（$q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ 与 $p(z) = \mathcal{N}(0, I)$ 之间）：

$$KL = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

代码中 `kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())` 精确对应此公式。

**重建项**（假设 $p_\theta(x|z)$ 为高斯分布）：

$$-\log p_\theta(x|z) \propto \|x - \hat{x}\|^2$$

代码中用 `MSELoss(reduction="sum")` 实现。

### 5. 总损失函数

$$\mathcal{L}_{VAE} = \underbrace{\sum_{i}(x_i - \hat{x}_i)^2}_{\text{重建损失}} \underbrace{- \frac{1}{2}\sum_{j}(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)}_{\text{KL 正则}}$$

两项的物理意义：
- **重建损失**：解码器能从 $z$ 准确还原 $x$
- **KL 正则**：编码器输出的分布接近标准正态，保证潜在空间的连续性和可采样性

### 6. 编码器网络的数学角色

代码中编码器输出两个向量：

| 层 | 输出 | 数学含义 |
|----|------|----------|
| `fc_mu` | $\mu_\phi(x)$ | 后验分布的均值 |
| `fc_logvar` | $\log \sigma_\phi^2(x)$ | 后验分布的对数方差（数值稳定性） |

使用 $\log \sigma^2$ 而非直接输出 $\sigma$ 的原因：$\sigma > 0$ 的约束通过 `exp()` 自动满足，且梯度更稳定。

### 7. 生成新样本

训练完成后，解码器可直接从先验采样生成：

$$z_{new} \sim \mathcal{N}(0, I), \quad \hat{x} = p_\theta(x|z_{new})$$

这是 VAE 相比普通自编码器的核心优势：潜在空间是规则的高斯分布，任意点都能解码为有效样本。

### 8. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `fc_mu(h)` | $\mu_\phi(x)$：编码器输出均值 |
| `fc_logvar(h)` | $\log \sigma_\phi^2(x)$：编码器输出对数方差 |
| `reparameterize(mu, logvar)` | $z = \mu + \sigma \cdot \epsilon$ |
| `MSELoss(reduction="sum")` | $-\log p_\theta(x|z) \propto \sum(x_i - \hat{x}_i)^2$ |
| `torch.sum(1 + logvar - mu^2 - logvar.exp())` | $KL(q_\phi \| p)$ 的闭式解 |
| `latent_dim=16` | 潜在空间维度 $d=16$ |
| `Sigmoid()` 输出层 | 输出归一化到 $[0,1]$，配合 MSE 重建损失 |
