"""
Q-Learning —— 经典的无模型（off-policy）时序差分强化学习算法
通过 Q 表格记录状态-动作价值，用 Bellman 方程迭代更新
"""
import numpy as np
from collections import defaultdict

# ===================== 1. 简单网格世界环境 =====================
class GridWorld:
    """4x4 网格世界：起点 (0,0)，终点 (3,3)，陷阱 (1,1) 和 (2,2)"""
    def __init__(self, size=4):
        self.size = size
        self.start = (0, 0)
        self.goal = (3, 3)
        self.traps = {(1, 1), (2, 2)}
        self.actions = [0, 1, 2, 3]  # 上、下、左、右
        self.action_names = ["↑", "↓", "←", "→"]
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 0: r = max(0, r - 1)  # 上
        elif action == 1: r = min(self.size - 1, r + 1)  # 下
        elif action == 2: c = max(0, c - 1)  # 左
        elif action == 3: c = min(self.size - 1, c + 1)  # 右

        self.state = (r, c)

        if self.state == self.goal:
            return self.state, 10.0, True  # 到达终点，奖励 +10
        elif self.state in self.traps:
            return self.state, -5.0, True  # 掉入陷阱，奖励 -5
        else:
            return self.state, -0.1, False  # 每步小惩罚，鼓励快速到达

    def render(self, q_table=None):
        symbols = np.full((self.size, self.size), ".", dtype=object)
        symbols[self.goal[0], self.goal[1]] = "G"
        for t in self.traps:
            symbols[t[0], t[1]] = "T"
        if q_table is not None:
            for r in range(self.size):
                for c in range(self.size):
                    if (r, c) not in self.traps and (r, c) != self.goal:
                        q_vals = [q_table.get(((r, c), a), 0) for a in self.actions]
                        symbols[r, c] = self.action_names[np.argmax(q_vals)]
        print("\n".join([" ".join(row) for row in symbols]))

# ===================== 2. Q-Learning 算法 =====================
def q_learning(env, n_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1, epsilon_decay=0.995):
    """Q-Learning 核心算法"""
    Q = defaultdict(float)  # Q(s,a) 默认值为 0
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # ε-greedy 探索策略
            if np.random.rand() < epsilon:
                action = np.random.choice(env.actions)
            else:
                q_vals = [Q[(state, a)] for a in env.actions]
                action = np.argmax(q_vals)

            next_state, reward, done = env.step(action)
            total_reward += reward

            # Q-Learning 更新（off-policy：用 max Q(s',a') 更新）
            best_next = max(Q[(next_state, a)] for a in env.actions)
            Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])

            state = next_state

        episode_rewards.append(total_reward)
        epsilon *= epsilon_decay  # 逐步降低探索率

    return Q, episode_rewards

# ===================== 3. 训练 =====================
env = GridWorld()
Q, rewards = q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.2)

print("=== Q-Learning 训练结果 ===")
print(f"最后 100 平均奖励: {np.mean(rewards[-100:]):.2f}")
print(f"收敛时奖励: {np.mean(rewards[-50:]):.2f}")

# ===================== 4. 策略展示 =====================
print("\n=== 学到的策略 ===")
env.render(Q)

# ===================== 5. Q 值表 =====================
print("\n=== Q 值表（部分）===")
print(f"{'状态':>10} {'↑':>8} {'↓':>8} {'←':>8} {'→':>8} {'最优':>6}")
for r in range(4):
    for c in range(4):
        state = (r, c)
        if state in env.traps or state == env.goal:
            continue
        q_vals = [Q[(state, a)] for a in env.actions]
        best = env.action_names[np.argmax(q_vals)]
        print(f"  ({r},{c}) {q_vals[0]:>8.3f} {q_vals[1]:>8.3f} {q_vals[2]:>8.3f} "
              f"{q_vals[3]:>8.3f}   {best}")

# ===================== 6. 超参数影响 =====================
print("\n=== 不同超参数对比 ===")
for alpha, gamma, eps in [(0.1, 0.99, 0.2), (0.5, 0.99, 0.2),
                           (0.1, 0.5, 0.2), (0.1, 0.99, 0.01)]:
    Q_test, r_test = q_learning(env, n_episodes=500, alpha=alpha, gamma=gamma, epsilon=eps)
    print(f"  α={alpha}, γ={gamma}, ε={eps}: 最后100平均奖励={np.mean(r_test[-100:]):.2f}")

print("\n=== Q-Learning 要点 ===")
print("- Off-policy：行为策略（ε-greedy）与目标策略（greedy）不同")
print("- Q(s,a) ← Q(s,a) + α × [r + γ×max Q(s',a') - Q(s,a)]")
print("- alpha: 学习率，越大更新越激进")
print("- gamma: 折扣因子，越大越重视远期奖励")
print("- epsilon: 探索率，越大越随机探索")
print("- 优点：简单、可收敛到最优 Q 值")
print("- 缺点：Q 表格大小 = 状态数 × 动作数，不适合连续状态/动作空间")
