# PPO：最流行的策略梯度算法

> 所属模块：05_强化学习 | 源文件：PPO.py | 核心功能：裁剪机制、GAE 优势估计、多 epoch 更新、OpenAI 标准实现

## 概述

PPO（Proximal Policy Optimization）是 OpenAI 2017 年提出的策略梯度算法，目前是强化学习领域**使用最广泛**的算法。它的成功来自两个特点：训练稳定性好 + 实现简单。

核心创新是**裁剪机制**：用概率比 r(theta) = pi_new/pi_old 衡量策略变化幅度，然后裁剪到 [1-epsilon, 1+epsilon] 范围内，防止策略更新过大导致崩溃。

脚本实现了完整的 PPO 训练流程：经验收集、GAE 优势估计、裁剪损失、多 epoch 更新。

## 关键代码解释

### 裁剪目标

`python
ratio = torch.exp(new_log_probs - old_log_probs)
clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
`

取 min 意味着：当策略变化方向正确时（A>0），最多让概率增加 epsilon 倍；方向错误时（A<0），最多让概率减少 epsilon 倍。

### GAE 优势估计

`python
delta = reward + gamma * V(next_state) - V(state)
gae = delta + gamma * lambda * gae
`

GAE 通过 lambda 参数平衡偏差和方差。lambda=0 退化为 TD，lambda=1 退化为蒙特卡洛。

## 使用示例

`python
agent = PPOAgent()
# 收集经验
for step in range(update_every):
    action, log_prob, value = agent.model.get_action(state)
# 更新
agent.update(states, actions, old_log_probs, returns, advantages)
`

## 注意事项

1. **clip_epsilon 通常取 0.1-0.3**
2. **多 epoch 复用数据**：同一批数据更新多次，提高样本效率
3. **GAE lambda 通常取 0.95**
4. **梯度裁剪**：clip_grad_norm 防止梯度爆炸

## 延伸思考

- **PPO for LLM**：ChatGPT 的 RLHF 阶段就使用了 PPO
- **PPO-Clip vs PPO-Penalty**：两种 PPO 变体
- **分布式 PPO**：如 IMPALA、Distributed PPO
- **Mamba 等替代方案**：状态空间模型能否替代 Transformer+PPO