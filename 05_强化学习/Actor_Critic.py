"""
Actor-Critic —— 结合策略梯度（Actor）和价值函数（Critic）的方法
Actor 负责选动作，Critic 负责评估状态价值，减少 REINFORCE 的方差
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

# ===================== 2. Actor-Critic 网络 =====================
class ActorCritic(nn.Module):
    """共享底层特征提取，分别输出策略（Actor）和价值（Critic）"""
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        # Actor: 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )
        # Critic: 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

# ===================== 3. Actor-Critic Agent =====================
class ACAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=3e-4, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state_t = torch.FloatTensor(state)
        probs, value = self.model(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def update(self, reward, log_prob, value, next_value, done):
        # 计算 TD 误差
        if done:
            target = reward
        else:
            target = reward + self.gamma * next_value.item()

        advantage = target - value.item()

        # Actor 损失：-log_prob × advantage
        actor_loss = -log_prob * advantage
        # Critic 损失：(target - value)²  — value 保持 tensor 参与运算
        target_t = torch.tensor(target, dtype=torch.float32)
        critic_loss = (target_t - value) ** 2

        loss = actor_loss + 0.5 * critic_loss  # 可调 critic 权重

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item()

# ===================== 4. 训练 =====================
env = SimpleCartPole()
agent = ACAgent()

print("=== Actor-Critic 训练 ===")
n_episodes = 500
reward_history = []

for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    # 收集一个 episode 的经验
    log_probs = []
    values = []
    rewards = []

    while not done:
        action, log_prob, value = agent.select_action(state)
        next_state, reward, done = env.step(action)

        # 用 TD(0) 在线更新（每步更新，不像 REINFORCE 等 episode 结束）
        next_state_t = torch.FloatTensor(next_state)
        with torch.no_grad():
            _, next_value = agent.model(next_state_t)

        agent.update(reward, log_prob, value, next_value, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        print(f"  Episode {episode+1:>3}: 平均奖励={avg:>6.1f}")

# ===================== 5. 测试 =====================
print("\n=== 测试 ===")
test_rewards = []
for _ in range(10):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        state_t = torch.FloatTensor(state)
        probs, _ = agent.model(state_t)
        action = probs.argmax().item()
        state, reward, done = env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)
print(f"10 次测试平均奖励: {np.mean(test_rewards):.1f}")

# ===================== 6. Actor-Critic 原理 =====================
print("\n=== Actor-Critic 原理 ===")
print("Actor（策略网络）: 输出 π(a|s)，决定选什么动作")
print("Critic（价值网络）: 输出 V(s)，评估当前状态有多好")
print("优势函数: A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)")
print("  - Actor 更新方向: 增加高优势动作的概率")
print("  - Critic 更新方向: 让 V(s) 更准确地预测回报")

print("\n=== Actor-Critic vs REINFORCE ===")
print("REINFORCE: 蒙特卡洛更新（等 episode 结束），高方差低偏差")
print("Actor-Critic: TD 更新（每步更新），低方差但有偏差（bootstrapping）")
print("  Actor-Critic 收敛更快、更稳定")

print("\n=== Actor-Critic 要点 ===")
print("- 结合了策略梯度和价值函数的优势")
print("- 每步更新（online learning），样本效率更高")
print("- 可扩展到连续动作空间（如 DDPG, SAC）")
print("- 基础版本仍有方差问题 → PPO/IMPALA 等进一步优化")
