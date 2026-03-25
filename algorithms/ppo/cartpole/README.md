# PPO (Proximal Policy Optimization) 算法

## 算法概述

PPO（近端策略优化）是 OpenAI 提出的策略梯度算法，通过裁剪策略更新幅度来保证训练稳定性。它是目前最流行、最实用的强化学习算法之一。

### 为什么需要 PPO？

```
传统策略梯度的问题：
┌─────────────────────────────────────────────────────────┐
│  新策略 πθ' 与旧策略 πθ 差异过大 → 性能崩塌            │
│                                                         │
│  回报                                                     │
│    │        ╱╲                                        │
│    │       ╱  ╲   ←← 大更新后性能崩溃                  │
│    │      ╱    ╲                                      │
│    │     ╱      ╲___                                  │
│    │    ╱                                          │
│    │   ╱                                           │
│    │  ╱                                            │
│    └──────────────────→ 训练步数                      │
│         ↑                                          │
│      这里策略变化太大！                               │
└─────────────────────────────────────────────────────────┘

PPO 的解决方案：
┌─────────────────────────────────────────────────────────┐
│  限制策略更新幅度：πθ'(a|s) / πθ(a|s) ∈ [1-ε, 1+ε]    │
│                                                         │
│  回报                                                     │
│    │                                            ╱╲     │
│    │                                          ╱  ╲    │
│    │                                        ╱    ╲   │
│    │                                      ╱      ╲  │
│    │                                    ╱          ╲│
│    └───────────────────────────────────────────────→ │
│                                                         │
│  平滑上升，没有崩溃！                                   │
└─────────────────────────────────────────────────────────┘
```

### 与其他算法对比

| 特性 | REINFORCE | Actor-Critic | PPO |
|-----|-----------|--------------|-----|
| 更新方式 | On-policy | On-policy | On-policy |
| 数据利用 | 1次/episode | 1次/episode | **多次/episode** |
| 稳定性 | 低 | 中 | **高** |
| 样本效率 | 低 | 中 | **高** |
| 实现难度 | 简单 | 中等 | 中等 |

---

## 核心公式

### 1. 重要性采样比率 (Importance Sampling Ratio)

衡量新策略与旧策略的差异：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

- $r_t > 1$: 新策略更倾向于该动作
- $r_t < 1$: 新策略更不倾向于该动作

### 2. PPO 裁剪目标 (Clipped Objective)

**未裁剪的目标**：
$$L^{CPI}(\theta) = \mathbb{E}\left[ r_t(\theta) \cdot A_t \right]$$

**裁剪后的目标**：
$$L^{CLIP}(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta) \cdot A_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \cdot A_t \right) \right]$$

其中 $\varepsilon$ 通常是 0.2。

**直观理解**：
```
当 A_t > 0（好动作）:
    如果 r_t > 1+ε: 限制更新，不要过度增加该动作概率
    否则: 正常更新

当 A_t < 0（坏动作）:
    如果 r_t < 1-ε: 限制更新，不要过度减少该动作概率
    否则: 正常更新
```

### 3. 价值函数损失

$$L^{VF}(\theta) = \mathbb{E}\left[ \left( V_\theta(s_t) - R_t \right)^2 \right]$$

### 4. 熵正则化 (Entropy Bonus)

鼓励探索，防止策略过早收敛：

$$L^{entropy}(\theta) = -\mathbb{E}\left[ \sum \pi_\theta(a|s) \log \pi_\theta(a|s) \right]$$

### 5. 总损失

$$L^{total}(\theta) = L^{CLIP}(\theta) + c_1 \cdot L^{VF}(\theta) - c_2 \cdot L^{entropy}(\theta)$$

其中 $c_1, c_2$ 是系数。

---

## GAE (Generalized Advantage Estimation)

GAE 是一种优势估计方法，平衡偏差和方差：

$$A_t^{GAE}(\gamma, \lambda) = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

其中 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 残差。

**简化实现**（从后向前计算）：
```python
advantages = []
last_advantage = 0
for t in reversed(range(len(rewards))):
    if t == len(rewards) - 1:
        next_value = 0
        next_non_terminal = 1 - done
    else:
        next_value = values[t + 1]
        next_non_terminal = 1 - done

    delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
    advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
    advantages.insert(0, advantage)
    last_advantage = advantage
```

**参数含义**：
- $\gamma = 0.99$: 折扣因子
- $\lambda = 0.95$: GAE 参数
  - $\lambda = 0$: 纯 TD（高偏差，低方差）
  - $\lambda = 1$: 蒙特卡洛（无偏，高方差）
  - $\lambda = 0.95$: 折中

---

## 算法伪代码

```
初始化策略网络 π(a|s; θ) 和价值网络 V(s; θ)
初始化优化器

for iteration = 1, 2, ..., K:
    # === 收集数据 ===
    for episode in n_episodes:
        state = env.reset()
        for t in each step:
            # 获取动作概率和价值
            action_probs, value = π(state), V(state)

            # 采样动作
            action ~ Categorical(action_probs)
            log_prob = log π(action|state)

            # 执行动作
            next_state, reward, done = env.step(action)

            # 存储数据
            buffer.add(state, action, reward, log_prob, value, done)
            state = next_state

            if done: break

    # === 计算 GAE 和回报 ===
    for each trajectory in buffer:
        advantages = compute_gae(rewards, values, dones)
        returns = advantages + values

    # === 多轮更新 ===
    for epoch in range(num_epochs):
        for mini_batch in buffer.shuffle().batch(batch_size):
            # 重新计算当前策略的概率
            new_probs = π(states)
            new_log_probs = log new_probs[actions]

            # 计算比率
            ratio = exp(new_log_probs - old_log_probs)

            # === 裁剪目标 ===
            surr1 = ratio * advantages
            surr2 = clip(ratio, 1-ε, 1+ε) * advantages
            policy_loss = -min(surr1, surr2).mean()

            # === 价值损失 ===
            value_pred = V(states)
            value_loss = MSE(value_pred, returns)

            # === 熵正则化 ===
            entropy = -sum(new_probs * log(new_probs))
            entropy_loss = -entropy.mean()

            # === 总损失 ===
            loss = policy_loss + c1 * value_loss + c2 * entropy_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新旧策略概率
        old_log_probs = new_log_probs.detach()

return π, V
```

---

## 算法流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        PPO 训练流程                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐                                                   │
│  │  开始   │                                                   │
│  └────┬────┘                                                   │
│       │                                                        │
│  ┌────▼─────┐    ┌─────────────────────────────────────────┐  │
│  │ 初始化   │───▶│ Actor-Critic 网络, 优化器                │  │
│  └────┬─────┘    └─────────────────────────────────────────┘  │
│       │                                                        │
│  ┌────▼──────────────────┐    ┌─────────────────────────────┐│
│  │  for iteration        │    │ 收集 n 个 episode 数据       ││
│  │  = 1 to K             │    │ 存储 (s, a, r, log_p, V)   ││
│  └────┬──────────────────┘    └──────────┬──────────────────┘│
│       │                                     │                 │
│       ▼                                     ▼                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │              计算 GAE 和回报                           │   │
│  │  A_t = δ_t + γλ(1-done)A_{t+1}                        │   │
│  │  R_t = A_t + V(s_t)                                    │   │
│  └────┬───────────────────────────────────────────────────┘   │
│       │                                                       │
│  ┌────▼─────────────────────┐    ┌─────────────────────────┐ │
│  │  for epoch = 1 to N      │    │ 多轮使用同一批数据       │ │
│  └────┬─────────────────────┘    └──────────┬──────────────┘ │
│       │                                      │               │
│  ┌────▼──────────────────────────────────────▼───────────┐  │
│  │                    计算损失                           │  │
│  │  ratio = π_new / π_old                               │  │
│  │  L_clip = -min(r·A, clip(r)·A)                       │  │
│  │  L_value = MSE(V, R)                                  │  │
│  │  L_entropy = -H(π)                                    │  │
│  └────┬───────────────────────────────────────────────────┘  │
│       │                                                      │
│  ┌────▼──────────────────────────────────────────────────┐  │
│  │                   反向传播更新                        │  │
│  └────┬──────────────────────────────────────────────────┘  │
│       │                                                      │
│  ┌────▼──────────────┐                                       │
│  │  达到最大迭代次数 │                                       │
│  │     结束训练      │                                       │
│  └───────────────────┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 实现要点

### 1. 网络架构

PPO 通常使用共享 backbone 的 Actor-Critic 网络：

```python
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor 头（输出动作概率）
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic 头（输出状态价值）
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = F.softmax(self.actor_head(features), dim=-1)
        value = self.critic_head(features)
        return action_probs, value

    def act(self, state):
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value
```

### 2. 经验回放缓冲区

```python
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get_batches(self, batch_size):
        # 返回打乱后的 mini-batches
        indices = np.random.permutation(len(self.states))
        for start in range(0, len(self.states), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield self._get_batch(batch_indices)
```

### 3. PPO 损失计算

```python
def compute_ppo_loss(self, states, actions, old_log_probs, advantages, returns):
    # 重新计算当前策略
    probs, values = self.network(states)
    dist = Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy()

    # 重要性采样比率
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 裁剪目标
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # 价值损失
    value_loss = F.mse_loss(values.squeeze(), returns)

    # 熵正则化
    entropy_loss = -entropy.mean()

    # 总损失
    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

    return loss, policy_loss, value_loss, entropy_loss
```

---

## 超参数建议

| 参数 | 建议值 | 说明 |
|-----|-------|------|
| `learning_rate` | 3e-4 | 学习率（PPO 通常用较小学习率） |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 参数 |
| `clip_epsilon` | 0.2 | 裁剪范围 |
| `entropy_coef` | 0.01 | 熵系数 |
| `value_coef` | 0.5 | 价值损失系数 |
| `num_epochs` | 4 | 每批数据更新轮数 |
| `batch_size` | 64 | mini-batch 大小 |
| `n_episodes_per_update` | 10-20 | 每次更新收集的 episode 数 |

---

## 优缺点

### 优点

- ✅ **稳定训练**: 裁剪机制防止策略更新过大
- ✅ **高样本效率**: 每批数据可以多次使用
- ✅ **易于实现**: 比 TRPO 简单很多
- ✅ **适用性广**: 连续和离散动作空间都适用
- ✅ **实践效果好**: 许多 SOTA 方法基于 PPO

### 缺点

- ❌ **仍需调参**: 超参数较多
- ❌ **收敛速度**: 相对较慢
- ❌ **局部最优**: 可能陷入局部最优

---

## 参考文献

1. Schulman, J., et al. (2017). **Proximal Policy Optimization Algorithms**. arXiv:1707.06347.
2. Schulman, J., et al. (2016). **High-Dimensional Continuous Control Using Generalized Advantage Estimation**. ICLR 2016.
