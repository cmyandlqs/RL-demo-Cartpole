# Actor-Critic 算法

## 算法概述

Actor-Critic（行动者-评论家）是结合策略梯度和值函数的强化学习算法。它通过引入 Critic 网络估计值函数，减少策略梯度的方差，提高样本效率。

### 与 REINFORCE 的对比

| 特性 | REINFORCE | Actor-Critic |
|-----|-----------|--------------|
| 网络结构 | 只有 Policy 网络 | Actor + Critic 两个网络 |
| 梯度估计 | 使用完整回报 G_t | 使用优势函数 A(s,a) |
| 方差 | 高 | 低（Critic 提供基线） |
| 偏差 | 无偏 | 有偏（Critic 估计误差） |
| 更新时机 | Episode 结束后 | 每步或每个 episode |

### 核心思想

```
┌─────────────────────────────────────────────────────────┐
│                    Actor-Critic 架构                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│    状态 s                                                │
│      │                                                  │
│      ├─────────────────────────────────────┐            │
│      │                                     │            │
│      ▼                                     ▼            │
│  ┌─────────┐                          ┌─────────┐      │
│  │  Actor  │                          │ Critic  │      │
│  │ π(a|s)  │                          │  V(s)   │      │
│  │         │                          │         │      │
│  │ 输出:   │                          │ 输出:   │      │
│  │ 动作概率│                          │ 状态价值│      │
│  └─────────┘                          └─────────┘      │
│      │                                     │            │
│      │ 动作 a                              │ 价值 V(s)  │
│      │                                     │            │
│      └─────────────────────────────────────┤            │
│                                             ▼            │
│                                    计算优势函数          │
│                                    A(s,a) = G_t - V(s)  │
│                                             │            │
│                                             ▼            │
│                                    更新 Actor 和 Critic│
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 核心公式

### 1. 优势函数 (Advantage Function)

优势函数衡量在状态 s 下采取动作 a 相对于平均水平的好坏：

$$A(s, a) = G_t - V(s)$$

其中：
- $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是折算回报
- $V(s)$ 是状态价值（Critic 输出）

**直观理解**：
- $A > 0$: 该动作比平均好 → 增加其概率
- $A < 0$: 该动作比平均差 → 减少其概率
- $V(s)$ 作为基线，不改变梯度方向但减少方差

### 2. Actor 损失

使用优势函数替代回报作为权重：

$$L_{actor} = -\mathbb{E}\left[ \log \pi(a|s; \theta) \cdot A(s, a) \right]$$

### 3. Critic 损失

Critic 网络学习准确的状态价值：

$$L_{critic} = \mathbb{E}\left[ \left( V(s) - G_t \right)^2 \right]$$

### 4. 总损失

$$L_{total} = L_{actor} + \alpha \cdot L_{critic}$$

其中 $\alpha$ 是平衡系数（通常取 0.5 或 1.0）。

---

## 算法伪代码

```
初始化 Actor 网络 π(a|s; θ) 和 Critic 网络 V(s; φ)
初始化优化器

for episode = 1, 2, ..., M:
    state = env.reset()
    trajectory = []  # 存储 (state, action, reward, log_prob, value)

    # === 采样轨迹 ===
    for t in each step:
        # 获取动作概率和状态价值
        action_probs = Actor(state)
        value = Critic(state)

        # 采样动作
        action ~ Categorical(action_probs)
        log_prob = log π(action|state)

        # 执行动作
        next_state, reward, done = env.step(action)

        # 记录数据
        trajectory.append((state, action, reward, log_prob, value.item()))
        state = next_state

        if done: break

    # === 计算回报和优势 ===
    for t in reversed(range(len(trajectory))):
        G_t = reward + gamma * G_t
        returns[t] = G_t
        # 优势 = 回报 - 价值估计
        advantages[t] = G_t - values[t]

    # 标准化优势（可选，有助于训练稳定）
    advantages = (advantages - mean) / (std + 1e-8)

    # === 更新网络 ===
    for each (state, action, log_prob, value) in trajectory:
        # Actor 损失（优势不参与梯度计算）
        actor_loss = -log_prob * advantages.detach()

        # Critic 损失
        critic_loss = MSE(value, returns)

        # 总损失
        loss = actor_loss + critic_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

return Actor, Critic
```

---

## 算法流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                      Actor-Critic 训练流程                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐                                                   │
│  │  开始   │                                                   │
│  └────┬────┘                                                   │
│       │                                                        │
│  ┌────▼─────┐    ┌─────────────────────────────────────────┐  │
│  │ 初始化   │───▶│ Actor: π(a|s;θ), Critic: V(s;φ)         │  │
│  │ 网络     │    └─────────────────────────────────────────┘  │
│  └────┬────┘                                                  │
│       │                                                       │
│  ┌────▼──────────────┐    ┌─────────────────────────────┐    │
│  │  for episode      │    │  state = env.reset()        │    │
│  │  = 1 to M         │    └──────────┬──────────────────┘    │
│  └────┬──────────────┘               │                       │
│       │                              │                       │
│  ┌────▼──────────────────────────────▼──────────────┐       │
│  │              采样一条轨迹                         │       │
│  │  action ~ π(s), value = V(s)                      │       │
│  │  存储 (s, a, r, log_π, V(s))                      │       │
│  └────┬──────────────────────────────────────────────┘       │
│       │                                                      │
│  ┌────▼──────────────────────────────────────────────┐       │
│  │            计算回报和优势函数                       │       │
│  │  G_t = r + γG_{t+1}                               │       │
│  │  A(s,a) = G_t - V(s)                              │       │
│  └────┬──────────────────────────────────────────────┘       │
│       │                                                      │
│  ┌────▼──────────────────────────────────────────────┐       │
│  │              计算损失并更新                        │       │
│  │  L_actor = -log π(a|s) · A(s,a)                   │       │
│  │  L_critic = MSE(V(s), G_t)                        │       │
│  │  L = L_actor + L_critic                           │       │
│  └────┬──────────────────────────────────────────────┘       │
│       │                                                      │
│  ┌────▼──────────────┐                                       │
│  │  达到最大回合数   │                                       │
│  │     结束训练      │                                       │
│  └───────────────────┘                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 实现要点

### 1. 网络架构

```python
# Actor: 策略网络
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

# Critic: 值函数网络
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出单个标量

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 输出 V(s)
```

### 2. 动作选择

```python
def select_action(self, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    # Actor 输出动作概率
    probs = self.actor(state_tensor)
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    # Critic 输出状态价值
    value = self.critic(state_tensor)

    return action.item(), log_prob, value
```

### 3. 优势函数计算

```python
def compute_advantages(self, rewards, values, gamma):
    advantages = []
    G_t = 0

    # 从后向前计算回报
    for r, v in zip(reversed(rewards), reversed(values)):
        G_t = r + gamma * G_t
        advantage = G_t - v.item()
        advantages.insert(0, advantage)

    advantages = torch.tensor(advantages, dtype=torch.float32)

    # 标准化（训练更稳定）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages
```

### 4. 网络更新

```python
def update(self, states, actions, log_probs, advantages, returns):
    # 重新计算当前策略的概率
    probs = self.actor(states)
    dist = Categorical(probs)
    new_log_probs = dist.log_prob(actions)

    # Actor 损失（优势不参与梯度）
    actor_loss = -(new_log_probs * advantages.detach()).mean()

    # Critic 损失
    values = self.critic(states).squeeze()
    critic_loss = F.mse_loss(values, returns)

    # 总损失
    loss = actor_loss + critic_loss

    # 反向传播
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

---

## 超参数建议

| 参数 | 建议值 | 说明 |
|-----|-------|------|
| `learning_rate` | 1e-3 ~ 5e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `hidden_dim` | 128 | 隐藏层维度 |
| `num_episodes` | 500 ~ 2000 | 训练回合数 |

---

## 优缺点

### 优点

- ✅ **方差小**: Critic 提供基线，减少方差
- ✅ **在线更新**: 不需要完整轨迹，可以每步更新
- ✅ **样本效率高**: 比 REINFORCE 更高效
- ✅ **适用性广**: 可处理连续和离散动作

### 缺点

- ❌ **有偏估计**: Critic 的估计误差会引入偏差
- ❌ **训练不稳定**: 两个网络需要同步训练
- ❌ **超参数敏感**: 需要平衡 Actor 和 Critic 的学习

---

## 改进方向

- **A2C / A3C**: 异步多线程训练
- **PPO**: 裁剪策略更新，稳定训练
- **DDPG**: 适用于连续动作空间
- **SAC**: Soft Actor-Critic，最大熵框架
