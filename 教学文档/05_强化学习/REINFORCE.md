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