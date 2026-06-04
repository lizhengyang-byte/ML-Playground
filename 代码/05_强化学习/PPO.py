"""
PPO（Proximal Policy Optimization）—— OpenAI 提出的稳定高效的策略梯度算法
核心：用裁剪机制限制策略更新幅度，防止灾难性更新
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ===================== 1. 简单 CartPole 环境 =====================
class SimpleCartPole:
    def __init__(self):
        self.max_steps = 200
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = 1.0 if action == 1 else -1.0
        x_ddot = (force - 0.0025 * x_dot + 0.001 * np.sin(theta)) / 1.0
        theta_ddot = (0.01 * np.sin(theta) - 0.001 * theta_dot + 0.0001 * force) / 0.1
        dt = 0.02
        x_dot += x_ddot * dt
        theta_dot += theta_ddot * dt
        x += x_dot * dt
        theta += theta_dot * dt
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        done = abs(x) > 2.4 or abs(theta) > 0.21 or self.steps >= self.max_steps
        reward = 1.0 if not done else 0.0
        return self.state.copy(), reward, done

# ===================== 2. PPO 网络 =====================
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_action(self, state):
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value

# ===================== 3. PPO 算法 =====================
class PPOAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=3e-4, gamma=0.99,
                 clip_epsilon=0.2, epochs=10, batch_size=64):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones, next_value):
        """广义优势估计（GAE）"""
        advantages = []
        gae = 0
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae  # lambda=0.95
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        """PPO 核心更新"""
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.epochs):
            # 随机打乱
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                probs, values = self.model(states[batch_idx])
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(actions[batch_idx])

                # PPO 裁剪目标
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                # Actor 损失（取较小值，保守更新）
                surr1 = ratio * advantages[batch_idx]
                surr2 = clipped_ratio * advantages[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic 损失
                critic_loss = nn.MSELoss()(values.squeeze(), returns[batch_idx])

                loss = actor_loss + 0.5 * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss

# ===================== 4. 训练 =====================
env = SimpleCartPole()
agent = PPOAgent()
update_every = 128  # 每收集 128 步更新一次

print("=== PPO 训练 ===")
n_episodes = 500
reward_history = []
total_steps = 0

# 收集经验
states_buf, actions_buf, rewards_buf, log_probs_buf, values_buf, dones_buf = [], [], [], [], [], []

for episode in range(n_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, value = agent.model.get_action(state_t)
        action = action.item()
        next_state, reward, done = env.step(action)

        states_buf.append(state)
        actions_buf.append(action)
        rewards_buf.append(reward)
        log_probs_buf.append(log_prob.item())
        values_buf.append(value.item())
        dones_buf.append(done)

        state = next_state
        episode_reward += reward
        total_steps += 1

        # 定期更新
        if total_steps % update_every == 0 and len(states_buf) >= update_every:
            with torch.no_grad():
                next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
                _, next_value = agent.model(next_state_t)
                next_value = next_value.item()

            advantages = agent.compute_gae(rewards_buf, values_buf, dones_buf, next_value)
            returns = [a + v for a, v in zip(advantages, values_buf)]

            agent.update(states_buf, actions_buf, log_probs_buf, returns, advantages)

            states_buf, actions_buf, rewards_buf = [], [], []
            log_probs_buf, values_buf, dones_buf = [], [], []

    reward_history.append(episode_reward)
    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"  Episode {episode+1:>3}: 平均奖励={avg:>6.1f}, 总步数={total_steps}")

# ===================== 5. 测试 =====================
print("\n=== 测试 ===")
test_rewards = []
for _ in range(10):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = agent.model(state_t)
        action = probs.argmax().item()
        state, reward, done = env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)
print(f"10 次测试平均奖励: {np.mean(test_rewards):.1f}")

# ===================== 6. PPO 核心机制 =====================
print("\n=== PPO 核心机制 ===")
print("裁剪目标: L = min(r(θ)×A, clip(r(θ), 1-ε, 1+ε)×A)")
print("  r(θ) = π_new(a|s) / π_old(a|s)  — 新旧策略的概率比")
print("  A = 优势函数（GAE 计算）")
print("  ε = 裁剪范围（通常 0.2）")
print()
print("裁剪的作用:")
print("- 当 A>0（好动作）: r 被裁剪到 1+ε，防止过度增加概率")
print("- 当 A<0（坏动作）: r 被裁剪到 1-ε，防止过度降低概率")
print("- 效果：策略更新幅度被限制，训练更稳定")

print("\n=== PPO 要点 ===")
print("- 结合了 TRPO 的稳定性和实现的简单性")
print("- clip_epsilon: 核心超参数，通常 0.1~0.3")
print("- epochs: 每批数据可复用多次，提高样本效率")
print("- GAE: 平衡偏差和方差的优势估计方法")
print("- 目前最流行的策略梯度算法之一（适用于连续和离散动作空间）")
