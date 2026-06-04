"""
SARSA —— On-policy 时序差分学习算法，与 Q-Learning 的区别在于更新时使用实际选择的下一动作
(S, A, R, S', A') → 走路名来源
"""
import numpy as np
from collections import defaultdict

# ===================== 1. 网格世界环境（同 Q_Learning.py）=====================
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.goal = (3, 3)
        self.traps = {(1, 1), (2, 2)}
        self.actions = [0, 1, 2, 3]
        self.action_names = ["↑", "↓", "←", "→"]

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 0: r = max(0, r - 1)
        elif action == 1: r = min(self.size - 1, r + 1)
        elif action == 2: c = max(0, c - 1)
        elif action == 3: c = min(self.size - 1, c + 1)
        self.state = (r, c)

        if self.state == self.goal:
            return self.state, 10.0, True
        elif self.state in self.traps:
            return self.state, -5.0, True
        return self.state, -0.1, False

    def render(self, q_table=None):
        symbols = np.full((self.size, self.size), ".", dtype=object)
        symbols[self.goal[0], self.goal[1]] = "G"
        for t in self.traps:
            symbols[t[0], t[1]] = "T"
        if q_table:
            for r in range(self.size):
                for c in range(self.size):
                    if (r, c) not in self.traps and (r, c) != self.goal:
                        q_vals = [q_table.get(((r, c), a), 0) for a in self.actions]
                        symbols[r, c] = self.action_names[np.argmax(q_vals)]
        print("\n".join([" ".join(row) for row in symbols]))

# ===================== 2. SARSA 算法 =====================
def sarsa(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995):
    """SARSA 核心算法（on-policy）"""
    Q = defaultdict(float)
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        # 选择第一个动作
        if np.random.rand() < epsilon:
            action = np.random.choice(env.actions)
        else:
            action = np.argmax([Q[(state, a)] for a in env.actions])

        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 选择下一个动作（SARSA 的关键：用行为策略选下一步动作）
            if np.random.rand() < epsilon:
                next_action = np.random.choice(env.actions)
            else:
                next_action = np.argmax([Q[(next_state, a)] for a in env.actions])

            # SARSA 更新（on-policy：用实际选择的 A' 更新）
            Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])

            state, action = next_state, next_action

        episode_rewards.append(total_reward)
        epsilon *= epsilon_decay

    return Q, episode_rewards

# ===================== 3. Q-Learning 对比 =====================
def q_learning(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995):
    Q = defaultdict(float)
    episode_rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice(env.actions)
            else:
                action = np.argmax([Q[(state, a)] for a in env.actions])
            next_state, reward, done = env.step(action)
            total_reward += reward
            best_next = max(Q[(next_state, a)] for a in env.actions)
            Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])
            state = next_state
        episode_rewards.append(total_reward)
        epsilon *= epsilon_decay
    return Q, episode_rewards

# ===================== 4. 训练与对比 =====================
env = GridWorld()
Q_sarsa, rewards_sarsa = sarsa(env, n_episodes=1000, epsilon=0.2)
Q_ql, rewards_ql = q_learning(env, n_episodes=1000, epsilon=0.2)

print("=== SARSA vs Q-Learning ===")
print(f"SARSA  最后 100 平均奖励: {np.mean(rewards_sarsa[-100:]):.2f}")
print(f"Q-Learning 最后 100 平均奖励: {np.mean(rewards_ql[-100:]):.2f}")

# ===================== 5. 策略对比 =====================
print("\n=== SARSA 策略 ===")
env.render(Q_sarsa)
print("\n=== Q-Learning 策略 ===")
env.render(Q_ql)

# ===================== 6. 关键区别 =====================
print("\n=== SARSA vs Q-Learning 核心区别 ===")
print("Q-Learning 更新: Q(s,a) += α × [r + γ × max Q(s',a') - Q(s,a)]")
print("SARSA 更新:      Q(s,a) += α × [r + γ × Q(s',a') - Q(s,a)]")
print("  其中 a' 是行为策略（ε-greedy）实际选择的动作，不是 greedy 的 max")
print()
print("效果差异:")
print("- Q-Learning 学到的是最优策略（冒险走捷径）")
print("- SARSA 学到的是更保守的策略（避开陷阱附近路径）")
print('- 因为 SARSA 在训练中"执行"了探索动作，更考虑探索风险')

print("\n=== SARSA 要点 ===")
print("- On-policy：行为策略 = 目标策略（都是 ε-greedy）")
print("- 比 Q-Learning 更保守（考虑探索时的风险）")
print("- 在有风险的环境中更安全（不走陷阱旁边）")
print("- 收敛速度可能比 Q-Learning 慢（因为受探索噪声影响）")
