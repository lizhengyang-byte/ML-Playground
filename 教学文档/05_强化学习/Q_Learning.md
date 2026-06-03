# Q-Learning：强化学习的入门课

> 所属模块：05_强化学习 | 源文件：Q_Learning.py | 核心功能：Off-policy 表格型 TD 学习、Q 表更新、超参数调优

## 概述

Q-Learning 是最经典的强化学习算法，属于无模型（model-free）、离策略（off-policy）的时序差分（TD）方法。它通过一张 Q 表记录每个状态-动作对的价值，用 Bellman 方程迭代更新。

核心更新公式：Q(s,a) <- Q(s,a) + alpha x [r + gamma x max Q(s',a') - Q(s,a)]

"Off-policy" 意味着行为策略（epsilon-greedy，用于探索）和目标策略（greedy，用于评估）是不同的。这让 Q-Learning 可以从任何经验中学习，不受当前探索策略的限制。

脚本在 4x4 网格世界中实现了完整的 Q-Learning 算法，展示了 Q 表、学到的策略和超参数影响。

## 关键代码解释

### Q 表用 defaultdict 存储

`python
Q = defaultdict(float)  # Q(s,a) 默认值为 0
`

用字典而非数组存储 Q 值，适合状态空间较大但大部分未访问的情况。

### Bellman 更新

`python
best_next = max(Q[(next_state, a)] for a in env.actions)
Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])
`

关键：max Q(s',a') —— 这就是 off-policy 的体现，取下一状态的最大 Q 值，不管实际选了什么动作。

## 使用示例

`python
Q = defaultdict(float)
state = env.reset()
action = epsilon_greedy(Q, state, epsilon)
next_state, reward, done = env.step(action)
best_next = max(Q[(next_state, a)] for a in env.actions)
Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])
`

## 注意事项

1. **Q 表大小 = 状态数 x 动作数**：连续空间不可行，需要 DQN
2. **epsilon 衰减**：初期多探索，后期多利用
3. **alpha 太大**：学习不稳定；太小：收敛太慢
4. **gamma 接近 1**：重视远期回报；接近 0：只看眼前

## 延伸思考

- **Double Q-Learning**：用两个 Q 表解决过估计问题
- **DQN**：用神经网络替代 Q 表，处理连续状态空间
- **Q-Learning 的收敛性**：在有限 MDP 下有理论保证，但实际中需要合理的学习率衰减
﻿## 数学原理

### 1. 马尔可夫决策过程（MDP）

强化学习的数学框架是 MDP，定义为五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$：

- $\mathcal{S}$：状态空间
- $\mathcal{A}$：动作空间
- $P(s'|s, a)$：状态转移概率
- $R(s, a)$：奖励函数
- $\gamma \in [0, 1)$：折扣因子

**代码对应**：`gymnasium` 环境封装了 MDP 的所有元素。

### 2. 值函数与 Bellman 方程

**状态值函数**：$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t | s_0 = s\right]$

**动作值函数**：$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t | s_0 = s, a_0 = a\right]$

**Bellman 方程**（最优）：

$$Q^*(s, a) = R(s, a) + \gamma\sum_{s'}P(s'|s, a)\max_{a'}Q^*(s', a')$$

### 3. Q-Learning 算法

**代码对应**：Q-Learning 是**离策略**（off-policy）时序差分学习。

更新规则：

$$Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma\max_{a'}Q(s', a') - Q(s, a)\right]$$

其中 $\alpha$ 为学习率，$r + \gamma\max_{a'}Q(s', a')$ 为 **TD 目标**。

关键：$\max_{a'}Q(s', a')$ 使用**贪婪策略**选择动作，而实际执行的动作由 $\epsilon$-贪婪策略决定（离策略）。

### 4. SARSA 算法

**代码对应**：SARSA 是**同策略**（on-policy）时序差分学习。

更新规则：

$$Q(s, a) \leftarrow Q(s, a) + \alpha\left[r + \gamma Q(s', a') - Q(s, a)\right]$$

其中 $a'$ 是**实际执行**的动作（由当前策略选择）。

### 5. Q-Learning vs SARSA

| 特性 | Q-Learning | SARSA |
|------|-----------|-------|
| 策略类型 | 离策略 | 同策略 |
| TD 目标 | $r + \gamma\max_{a'}Q(s', a')$ | $r + \gamma Q(s', a')$ |
| 收敛到 | $Q^*$（最优 Q 函数） | $Q^\pi$（当前策略的 Q 函数） |
| 风险偏好 | 更激进（学习最优） | 更保守（考虑探索风险） |

**代码对应**：Q-Learning 更新中 `np.max(Q[next_state])` vs SARSA 中 `Q[next_state, next_action]`。

### 6. $\epsilon$-贪婪策略

$$\pi(a|s) = \begin{cases} 1 - \epsilon + \epsilon/|\mathcal{A}| & a = \arg\max_{a'}Q(s, a') \\ \epsilon/|\mathcal{A}| & \text{otherwise} \end{cases}$$

$\epsilon$ 控制探索与利用的平衡。通常从 $\epsilon = 1$ 逐步衰减到 $\epsilon_{\min}$。
