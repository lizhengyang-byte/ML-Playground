"""
DQN（Deep Q-Network）—— 用神经网络近似 Q 函数，解决 Q-Learning 无法处理连续状态空间的问题
引入经验回放（Experience Replay）和目标网络（Target Network）稳定训练
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ===================== 1. 简单 CartPole 环境（自实现）=====================
class SimpleCartPole:
    """简化版 CartPole 环境，状态: [位置, 速度, 角度, 角速度]"""
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
        # 物理模拟（简化）
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

# ===================== 2. DQN 网络 =====================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# ===================== 3. 经验回放 =====================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# ===================== 4. DQN Agent =====================
class DQNAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=32, target_update=10):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.losses = []

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self.q_net(state_t)
            return q_vals.argmax(dim=1).item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # 当前 Q 值
        q_values = self.q_net(states_t).gather(1, actions_t).squeeze()

        # 目标 Q 值（使用目标网络）
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ===================== 5. 训练 =====================
env = SimpleCartPole()
agent = DQNAgent()

print("=== DQN 训练 ===")
n_episodes = 300
reward_history = []

for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    if (episode + 1) % agent.target_update == 0:
        agent.update_target()

    reward_history.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg = np.mean(reward_history[-50:])
        avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
        print(f"  Episode {episode+1:>3}: 平均奖励={avg:>6.1f}, "
              f"ε={agent.epsilon:.3f}, 平均Loss={avg_loss:.4f}")

# ===================== 6. 测试 =====================
print("\n=== 测试（关闭探索）===")
agent.epsilon = 0
test_rewards = []
for _ in range(10):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.select_action(state)
        state, reward, done = env.step(action)
        total_reward += reward
    test_rewards.append(total_reward)
print(f"10 次测试平均奖励: {np.mean(test_rewards):.1f}")

# ===================== 7. DQN 关键组件 =====================
print("\n=== DQN 关键组件 ===")
print("1. 经验回放 (Experience Replay): 打破数据相关性，提高样本效率")
print("2. 目标网络 (Target Network): 冻结目标 Q 值，防止训练振荡")
print("3. ε-greedy 探索: 初期大量探索，逐步趋向利用")
print("4. 神经网络近似: 用函数 Q(s,a;θ) 近似 Q 表格，处理连续状态空间")

print("\n=== DQN 要点 ===")
print("- 解决了 Q-Learning 在连续状态空间的局限")
print("- 训练不稳定：需要目标网络、经验回放、梯度裁剪等技巧")
print("- 只适合离散动作空间（连续动作用 DDPG/PPO）")
print("- Atari 游戏中表现惊人（仅用像素输入）")
