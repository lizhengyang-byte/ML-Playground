# DQN：深度 Q 网络——从 Q 表到神经网络的跨越
> 难度标签：高级 | 预计时长：20-30分钟 | 前置知识：无学习经验


> 所属模块：05_强化学习 | 源文件：DQN.py | 核心功能：经验回放、目标网络、epsilon 衰减、PyTorch 实现

## 概述

DQN（Deep Q-Network）是 DeepMind 2015 年在 Nature 上发表的里程碑工作。它用神经网络替代 Q 表，配合经验回放和目标网络两大技巧，在 Atari 游戏上达到了超越人类的水平。

Q-Learning 的 Q 表在连续状态空间中不可行（状态数无穷多），DQN 用一个神经网络 Q(s,a;theta) 来近似 Q 函数。

脚本在自实现的 CartPole 环境中训练 DQN，展示了经验回放、目标网络和训练过程。

## 关键代码解释

### 两大创新

**经验回放**：打破时间相关性，让数据可以复用。

```python
buffer.push(state, action, reward, next_state, done)
batch = buffer.sample(batch_size)  # 随机采样一批经验
```

**目标网络**：冻结目标 Q 值，防止训练振荡。

```python
target = rewards + gamma * target_net(next_states).max(dim=1)[0] * (1 - dones)
# 每隔 target_update 步同步一次
target_net.load_state_dict(q_net.state_dict())
```

### 损失函数

```python
q_values = q_net(states).gather(1, actions)
loss = MSELoss(q_values, target)
```

## 使用示例

```python
agent = DQNAgent(state_dim=4, action_dim=2)
for episode in range(n_episodes):
    state = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
```

## 注意事项

1. **只适合离散动作**：连续动作空间用 DDPG/SAC
2. **训练不稳定**：需要目标网络、经验回放、梯度裁剪
3. **样本效率低**：需要大量交互数据
4. **过估计偏差**：max 操作会高估 Q 值，Double DQN 可缓解

## 延伸思考

- **Double DQN**：用在线网络选动作，目标网络评估，减少过估计
- **Dueling DQN**：将 Q 值分解为 V(s) + A(s,a)
- **Prioritized Experience Replay**：优先采样 TD 误差大的经验
- **Rainbow DQN**：融合多种 DQN 变体的集大成者
## 数学原理

### 1. DQN 的核心思想

**代码对应**：DQN 用神经网络 $Q(s, a; \theta)$ 近似 Q 函数。

Q-Learning 的表格法在状态空间大或连续时不可行。DQN 用深度网络参数化 Q 函数：

$$Q(s, a; \theta) \approx Q^*(s, a)$$

### 2. 损失函数

$$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\left[\left(r + \gamma\max_{a'}Q(s', a'; \theta^{-}) - Q(s, a; \theta)\right)^2\right]$$

其中 $\theta^{-}$ 为目标网络参数（定期从 $\theta$ 复制），$\mathcal{D}$ 为经验回放缓冲区。

### 3. 经验回放（Experience Replay）

**代码对应**：经验回放打破数据间的时间相关性。

存储转移 $(s, a, r, s')$ 到缓冲区 $\mathcal{D}$，训练时随机采样 mini-batch：

$$\theta \leftarrow \theta - \alpha\nabla_\theta L(\theta)$$

梯度：

$$\nabla_\theta L = \mathbb{E}\left[2(r + \gamma\max_{a'}Q(s', a'; \theta^-) - Q(s, a; \theta))\nabla_\theta Q(s, a; \theta)\right]$$

### 4. 目标网络（Target Network）

直接用 $Q(s', a'; \theta)$ 计算 TD 目标会导致"追逐移动目标"的不稳定。目标网络 $\theta^-$ 定期从 $\theta$ 复制：

$$y = r + \gamma\max_{a'}Q(s', a'; \theta^-)$$

这使训练目标在一定时期内保持稳定。

### 5. $\epsilon$-贪婪探索

$$a = \begin{cases}\arg\max_a Q(s, a; \theta) & \text{with prob } 1-\epsilon \\ \text{random action} & \text{with prob } \epsilon\end{cases}$$

$\epsilon$ 通常从 1.0 线性衰减到 0.01。
