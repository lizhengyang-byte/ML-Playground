# Actor-Critic：策略梯度与价值函数的联姻

> 所属模块：05_强化学习 | 源文件：Actor_Critic.py | 核心功能：双网络架构、TD 更新、优势函数、共享特征

## 概述

Actor-Critic 结合了 REINFORCE（策略梯度）和价值函数的优势。Actor 负责选动作（策略网络），Critic 负责评估状态价值（价值网络）。

REINFORCE 用蒙特卡洛回报 G_t 更新，方差大。Actor-Critic 用 TD 误差（Advantage）替代 G_t，大幅降低方差——代价是引入偏差（bootstrapping）。

脚本用共享底层特征的 Actor-Critic 网络在 CartPole 上训练。

## 关键代码解释

### 优势函数

`python
advantage = reward + gamma * next_value - value  # TD 误差
`

优势函数 A(s,a) = Q(s,a) - V(s) 衡量"这个动作比平均水平好多少"。正优势增加概率，负优势降低概率。

### 双头网络

`python
self.actor = ...   # 输出动作概率
self.critic = ...  # 输出状态价值 V(s)
`

共享底层特征提取，减少参数量。

## 注意事项

1. **TD 更新 vs MC 更新**：每步更新（低方差有偏差）vs episode 结束更新（高方差无偏差）
2. **学习率敏感**：Actor 和 Critic 可能需要不同学习率
3. **训练不稳定**：基础版本仍有问题，需要 PPO 等改进

## 延伸思考

- **A2C（Advantage Actor-Critic）**：同步并行版本
- **A3C（Asynchronous Advantage Actor-Critic）**：异步并行版本
- **GAE（Generalized Advantage Estimation）**：平衡偏差和方差的通用优势估计
- **DDPG**：连续动作空间的 Actor-Critic
﻿## 数学原理

### 1. Actor-Critic 架构

**代码对应**：Actor-Critic 结合策略梯度（Actor）和值函数（Critic）。

**Actor**（策略网络）：$\pi_\theta(a|s)$，输出动作概率分布

**Critic**（值网络）：$V_\phi(s)$，估计状态值函数

### 2. Actor 的更新

用 **TD 误差** $\delta_t$ 代替蒙特卡洛回报 $G_t$：

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

Actor 更新：

$$\theta \leftarrow \theta + \alpha_\theta\sum_t\nabla_\theta\ln\pi_\theta(a_t|s_t) \cdot \delta_t$$

### 3. Critic 的更新

Critic 通过最小化 TD 误差的平方更新：

$$\phi \leftarrow \phi - \alpha_\phi\nabla_\phi\sum_t\delta_t^2$$

$$\nabla_\phi\delta_t^2 = 2\delta_t\left(-\nabla_\phi V_\phi(s_t)\right)$$

### 4. 优势函数

TD 误差是优势函数 $A(s, a)$ 的无偏估计：

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \approx r_t + \gamma V(s_{t+1}) - V(s_t) = \delta_t$$

$A > 0$：动作比平均好（鼓励），$A < 0$：动作比平均差（抑制）。

### 5. REINFORCE vs Actor-Critic

| 特性 | REINFORCE | Actor-Critic |
|------|----------|-------------|
| 回报估计 | 蒙特卡洛 $G_t$ | TD 误差 $\delta_t$ |
| 偏差 | 无偏 | 有偏（Critic 不完美） |
| 方差 | 高 | 低 |
| 更新时机 | 轨迹结束后 | 每步更新 |
| 收敛速度 | 慢 | 快 |
