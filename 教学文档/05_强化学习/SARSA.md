# SARSA：更谨慎的表格型强化学习

> 所属模块：05_强化学习 | 源文件：SARSA.py | 核心功能：On-policy TD 学习、与 Q-Learning 对比、安全策略

## 概述

SARSA（State-Action-Reward-State-Action）是 on-policy 的 TD 学习算法。名字来源于它的更新需要五元组 (S, A, R, S', A')。

与 Q-Learning 的核心区别：Q-Learning 用 max Q(s',a') 更新（假想中的最优动作），SARSA 用实际选择的 Q(s',a') 更新。这让 SARSA 更保守——它在训练中考虑了探索的风险。

## 关键代码解释

### SARSA 更新

`python
# 先选下一个动作，再更新
next_action = epsilon_greedy(...)
Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])
`

SARSA 必须在更新前就选好下一步动作 A'，因为更新需要用到实际选的 A'。

### 策略差异

在有陷阱的环境中，Q-Learning 学到的策略会走陷阱旁边（因为最优路径更短），SARSA 学到的策略会远离陷阱（因为探索时可能掉进去）。

## 注意事项

1. **On-policy**：训练数据来自当前策略，策略改变后旧数据不可用
2. **更安全**：在有风险的环境中比 Q-Learning 更保守
3. **收敛更慢**：受探索噪声影响

## 延伸思考

- **Expected SARSA**：用 Q 值的期望替代实际采样的 A'，减少方差
- **SARSA(lambda)**：资格迹（Eligibility Trace），加速信用分配
- **安全 RL**：SARSA 的保守性启发了约束策略优化（CPO）
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
