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