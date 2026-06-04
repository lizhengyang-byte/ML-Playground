# GAN：生成器与判别器的博弈
> 难度标签：高级 | 预计时长：15-25分钟 | 前置知识：无学习经验


> 所属模块：06_深度学习 | 源文件：生成对抗网络_GAN.py | 核心功能：对抗训练、生成器/判别器、模式崩溃

## 概述

GAN 由生成器 G 和判别器 D 组成。G 试图生成以假乱真的样本，D 试图区分真假。两者博弈训练，最终 G 能生成逼真数据。

## 关键代码解释

### 交替训练

```python
# 判别器：区分真假
d_loss = -torch.mean(torch.log(D(real)) + torch.log(1 - D(G(noise))))
# 生成器：骗过判别器
g_loss = -torch.mean(torch.log(D(G(noise))))
```

### 训练技巧

```python
# 实际中用 BCE loss 替代原始公式
# 标签平滑：real_label = 0.9
# 噪声输入：给判别器输入加噪声
```

## 注意事项

1. **训练不稳定**：GAN 训练是出了名的难
2. **模式崩溃**：生成器只生成少数几种样本
3. **评估困难**：没有像 loss 这样直观的评估指标

## 延伸思考

- **WGAN**：用 Wasserstein 距离替代 JS 散度，训练更稳定
- **StyleGAN**：高质量图像生成
- **扩散模型**：正在取代 GAN 成为主流生成模型
## 数学原理

### 1. GAN 的博弈论框架

GAN 由生成器 $G$ 和判别器 $D$ 组成，形成一个**极小极大博弈**：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

- $D(x)$：判别器判断 $x$ 为真实数据的概率
- $G(z)$：生成器从噪声 $z \sim p_z(z)$ 生成的假样本
- $D$ 想最大化 $V$（正确区分真假），$G$ 想最小化 $V$（骗过 $D$）

### 2. 判别器的最优解

固定 $G$ 时，最优判别器为：

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

其中 $p_g$ 是生成器隐式定义的数据分布。直觉：当 $p_{data}(x) = p_g(x)$ 时，$D^*(x) = 0.5$，即判别器完全无法区分。

### 3. 全局最优与 JS 散度

将 $D^*$ 代入目标函数，得到：

$$C(G) = -\log 4 + 2 \cdot JSD(p_{data} \| p_g)$$

其中 $JSD$ 是 **Jensen-Shannon 散度**：

$$JSD(P \| Q) = \frac{1}{2} KL(P \| M) + \frac{1}{2} KL(Q \| M), \quad M = \frac{P + Q}{2}$$

当且仅当 $p_g = p_{data}$ 时，$JSD = 0$，$C(G)$ 取最小值 $-\log 4$。

### 4. 判别器损失

实际训练中，判别器的损失函数为：

$$\mathcal{L}_D = -\frac{1}{m}\sum_{i=1}^{m}\left[\log D(x^{(i)}) + \log(1 - D(G(z^{(i)})))\right]$$

对应的代码中 `criterion = nn.BCELoss()` 实现了二元交叉熵。

### 5. 生成器损失

原始论文的生成器损失：

$$\mathcal{L}_G = \frac{1}{m}\sum_{i=1}^{m}\log(1 - D(G(z^{(i)})))$$

实践中常使用**非饱和启发**（non-saturating heuristic）以避免梯度消失：

$$\mathcal{L}_G^{ns} = -\frac{1}{m}\sum_{i=1}^{m}\log D(G(z^{(i)}))$$

即让生成器最大化"判别器把假样本判为真"的概率。

### 6. 训练动态分析

**训练平衡**：$D$ 和 $G$ 需要交替训练，保持能力平衡。

| 问题 | 表现 | 数学原因 |
|------|------|----------|
| $D$ 太强 | $D(G(z)) \approx 0$，$G$ 梯度消失 | $\log(1-D(G(z)))$ 饱和 |
| $G$ 太强 | 模式崩塌（mode collapse） | $G$ 只学会生成少数样本 |
| 理想平衡 | $D(x) \approx 0.5$ | $p_g \approx p_{data}$ |

代码中 `betas=(0.5, 0.999)` 是 GAN 特有的 Adam 参数，一阶动量降低以稳定训练。

### 7. 重参数化与采样

生成器的输入噪声通常采样自：

$$z \sim \mathcal{N}(0, I) \quad \text{或} \quad z \sim \text{Uniform}(-1, 1)$$

代码中 `latent_dim=32` 对应噪声向量维度。生成器学习的是一个从低维噪声空间到高维数据空间的映射 $G: \mathcal{Z} \to \mathcal{X}$。

### 8. 代码与数学的对应

| 代码 | 数学含义 |
|------|----------|
| `Generator(latent_dim=32)` | $G(z): \mathbb{R}^{32} \to \mathbb{R}^{64}$ |
| `Discriminator(input_dim=64)` | $D(x): \mathbb{R}^{64} \to [0,1]$ |
| `BCELoss` 用于 $D$ | $\mathcal{L}_D = -[\log D(x) + \log(1-D(G(z)))]$ |
| `BCELoss` 用于 $G$（标签=1） | $\mathcal{L}_G^{ns} = -\log D(G(z))$ |
| `Tanh()` 输出层 | 将生成样本归一化到 $[-1, 1]$ |
| `LeakyReLU(0.2)` | 避免判别器梯度稀疏，$\text{LReLU}(x) = \max(0.2x, x)$ |
| `betas=(0.5, 0.999)` | 降低 Adam 一阶动量以稳定 GAN 训练 |
