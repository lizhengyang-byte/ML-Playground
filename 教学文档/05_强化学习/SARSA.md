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