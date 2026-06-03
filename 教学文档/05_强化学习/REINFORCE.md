# REINFORCE：最朴素的策略梯度算法

> 所属模块：05_强化学习 | 源文件：REINFORCE.py | 核心功能：蒙特卡洛策略梯度、回报标准化、PyTorch 策略网络

## 概述

REINFORCE 是最基础的策略梯度算法。与 Q-Learning/SARSA 学习价值函数不同，REINFORCE 直接优化策略函数 pi(a|s;theta)。

核心思想：如果一个动作带来了高回报，就增加它被选中的概率；反之则降低。数学上基于策略梯度定理：梯度 J(theta) = E[梯度 log pi(a|s) x G_t]。

REINFORCE 是蒙特卡洛方法——必须等整个 episode 结束才能计算回报 G_t 并更新。这导致高方差，但低偏差。

## 关键代码解释

### 折扣回报计算

`python
G = 0
for r in reversed(self.rewards):
    G = r + self.gamma * G
    returns.insert(0, G)
`

从后往前累加折扣回报。这是蒙特卡洛方法的标志——用完整的 episode 回报。

### 回报标准化

`python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
`

标准化减少方差，是 REINFORCE 的标准实践。

### 策略梯度损失

`python
loss -= log_prob * G  # 增加高回报动作的概率
`

## 注意事项

1. **高方差**：蒙特卡洛采样方差大，需要很多 episode 才能收敛
2. **On-policy**：每次更新后旧数据作废
3. **回报标准化很重要**：否则训练极不稳定

## 延伸思考

- **加 baseline 的 REINFORCE**：用 V(s) 做 baseline 减少方差，这就引出了 Actor-Critic
- **Entropy Bonus**：鼓励探索，防止策略过早收敛
- **VPG（Vanilla Policy Gradient）**：REINFORCE 的通用框架版本
﻿## 数学原理

### 1. 策略梯度定理

**代码对应**：REINFORCE 直接优化策略 $\pi(a|s; \theta)$。

目标函数：$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]$

**策略梯度定理**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\nabla_\theta\ln\pi_\theta(a_t|s_t) \cdot G_t\right]$$

其中 $G_t = \sum_{k=t}^{\infty}\gamma^{k-t}r_k$ 为从时刻 $t$ 开始的**回报**（return）。

### 2. REINFORCE 算法

**蒙特卡洛策略梯度**：用采样轨迹估计期望：

$$\theta \leftarrow \theta + \alpha\sum_{t=0}^{T}\nabla_\theta\ln\pi_\theta(a_t|s_t) \cdot G_t$$

流程：
1. 用当前策略 $\pi_\theta$ 采样完整轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$
2. 计算每个时刻的回报 $G_t$
3. 更新 $\theta$

### 3. 基线（Baseline）减小方差

**代码对应**：引入基线 $b(s)$（如状态值函数 $V(s)$）减小方差：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta\ln\pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

$b(s)$ 不改变梯度的期望（无偏），但显著减小方差。常用 $b(s) = V(s; \phi)$。

### 4. 优缺点

- **优点**：可处理连续动作空间，策略梯度是无偏的
- **缺点**：方差大（蒙特卡洛采样），收敛慢
- **改进**：Actor-Critic 用 TD 误差代替蒙特卡洛回报，减小方差
