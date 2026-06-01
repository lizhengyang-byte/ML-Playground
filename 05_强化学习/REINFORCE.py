"""
REINFORCE —— 基于策略梯度的蒙特卡洛方法，直接优化策略函数
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

# ===================== 2. 策略网络 =====================
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)

# ===================== 3. REINFORCE 算法 =====================
class REINFORCEAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        state_t = torch.FloatTensor(state)
        probs = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        # 计算折扣回报
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # 标准化回报（减少方差）
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 策略梯度损失: -E[log π(a|s) × G]
        loss = 0
        for log_prob, G in zip(self.log_probs, returns):
            loss -= log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

# ===================== 4. 训练 =====================
env = SimpleCartPole()
agent = REINFORCEAgent()

print("=== REINFORCE 训练 ===")
n_episodes = 500
reward_history = []

for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        agent.store_reward(reward)
        total_reward += reward

    agent.update()
    reward_history.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"  Episode {episode+1:>3}: 平均奖励={avg:>6.1f}")

# ===================== 5. 测试 =====================
print("\n=== 测试（无探索）===")
test_rewards = []
for _ in range(10):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        state_t = torch.FloatTensor(state)
        probs = agent.policy(state_t)
        action = probs.argmax().item()
        state, reward, done = env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)
print(f"10 次测试平均奖励: {np.mean(test_rewards):.1f}")

# ===================== 6. REINFORCE 原理 =====================
print("\n=== REINFORCE 原理 ===")
print("策略梯度定理: ∇J(θ) = E[∇log π(a|s;θ) × G_t]")
print("  π(a|s;θ): 给定状态 s 下选择动作 a 的概率")
print("  G_t: 从 t 时刻起的折扣累积回报")
print("  更新方向：增加高回报动作的概率，降低低回报动作的概率")

print("\n=== REINFORCE 要点 ===")
print("- On-policy：需要从当前策略采样数据")
print("- 蒙特卡洛方法：等 episode 结束后才更新（高方差）")
print("- 回报标准化可减少方差：(G - mean(G)) / std(G)")
print("- 优点：可处理连续动作空间、理论保证收敛")
print("- 缺点：高方差、收敛慢（相比 Actor-Critic）")
print("- 改进：加 baseline（如价值函数）减少方差 → Actor-Critic")
