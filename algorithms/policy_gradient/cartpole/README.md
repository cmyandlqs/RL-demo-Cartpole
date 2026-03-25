# Policy Gradient (REINFORCE) 算法

## 算法概述

Policy Gradient 是一类直接参数化策略函数 π(a|s;θ)，通过梯度上升直接优化期望回报的强化学习算法。REINFORCE 是最基础的 Policy Gradient 算法，也称为蒙特卡洛策略梯度。

### 与 DQN 的区别

| 特性 | DQN | Policy Gradient |
|-----|-----|-----------------|
| 学习方式 | 学习 Q(s,a) 值函数 | 直接学习策略 π(a\|s) |
| 网络输出 | 每个 action 的 Q 值 | action 的概率分布 |
| 适用场景 | 离散动作空间 | 离散/连续动作空间 |
| 探索机制 | ε-greedy | 固有随机性（概率分布） |

---

## 核心公式

### 1. 目标函数

我们希望最大化期望回报：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

其中：
- $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ 是一条轨迹
- $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$ 是轨迹的折算回报

### 2. 策略梯度定理

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi(a|s;\theta) \cdot G_t \right]$$

其中：
- $\pi(a|s;\theta)$ 是策略网络，输出在状态 s 下采取动作 a 的概率
- $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从时刻 t 开始的折算回报
- $\log \pi(a|s;\theta)$ 是对数概率

**直观理解**：
- 如果 $G_t > 0$（回报为正），增加该动作的概率
- 如果 $G_t < 0$（回报为负），减少该动作的概率
- $G_t$ 越大，概率调整幅度越大

### 3. 损失函数

为了最大化目标函数，我们最小化负对数似然损失：

$$L(\theta) = -\mathbb{E}\left[ \log \pi(a_t|s_t;\theta) \cdot G_t \right]$$

实际实现中：

```python
# 对于轨迹中的每个时间步
loss = -log_prob[action] * G_t
total_loss = loss.mean()
```

### 4. 折算回报计算

从时刻 t 开始的折算回报：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T$$

```python
# 从后向前计算
G_t = 0
for t in reversed(range(len(rewards))):
    G_t = rewards[t] + gamma * G_t
    returns[t] = G_t
```

---

## 算法流程

### REINFORCE 算法伪代码

```
初始化策略网络参数 θ 随机

for episode = 1, 2, ..., M:
    # 1. 采样轨迹
    state = env.reset()
    trajectory = []  # 存储 (state, action, log_prob, reward)

    for t in each step:
        # 2. 策略网络输出动作概率分布
        action_probs = π(state; θ)

        # 3. 根据概率分布采样动作
        action ~ Categorical(action_probs)

        # 4. 执行动作，获得反馈
        next_state, reward, done = env.step(action)

        # 5. 记录 (state, action, log_prob, reward)
        trajectory.append((state, action, log_prob, reward))
        state = next_state

        if done: break

    # 6. 计算折算回报 G_t
    for t in reversed(range(len(trajectory))):
        G_t = reward + gamma * G_t
        returns[t] = G_t

    # 7. 计算梯度并更新
    loss = -mean(log_prob * G_t)  # 负号因为要最大化
    θ = θ - α * ∇_θ loss

return θ
```

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    REINFORCE 算法流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────┐    │
│  │  开始   │───→│ 初始化网络 θ │───→│  for episode   │    │
│  └─────────┘    └─────────────┘    └────────┬────────┘    │
│                                            │               │
│                                    ┌───────▼───────┐      │
│                                    │  采样一条轨迹  │      │
│                                    │  π(a|s;θ)     │      │
│                                    └───────┬───────┘      │
│                                            │               │
│                                    ┌───────▼───────┐      │
│                                    │ 计算 G_t（从后 │      │
│                                    │ 向前累加奖励） │      │
│                                    └───────┬───────┘      │
│                                            │               │
│                                    ┌───────▼───────┐      │
│                                    │ 计算损失       │      │
│                                    │ L = -logπ·G_t │      │
│                                    └───────┬───────┘      │
│                                            │               │
│                                    ┌───────▼───────┐      │
│                                    │ 反向传播更新 θ │      │
│                                    └───────┬───────┘      │
│                                            │               │
│                                    ┌───────▼───────┐      │
│                                    │ 达到最大回合数 │      │
│                                    │    结束训练    │      │
│                                    └───────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 实现要点

### 1. 策略网络结构

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)  # 输出概率分布
```

### 2. 动作选择

```python
def select_action(self, state):
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    probs = self.network(state_tensor)  # 获取概率分布

    # 从概率分布中采样（探索的来源）
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.item(), log_prob
```

### 3. 回报计算

```python
def compute_returns(self, rewards, gamma):
    returns = []
    G_t = 0
    for r in reversed(rewards):
        G_t = r + gamma * G_t
        returns.insert(0, G_t)
    returns = torch.tensor(returns)
    # 标准化（可选，有助于训练稳定）
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    return returns
```

### 4. 训练更新

```python
def update(self, log_probs, returns):
    # 计算损失
    loss = []
    for log_prob, G_t in zip(log_probs, returns):
        loss.append(-log_prob * G_t)
    loss = torch.stack(loss).sum()

    # 反向传播
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

---

## 超参数建议

| 参数 | 建议值 | 说明 |
|-----|-------|------|
| `learning_rate` | 1e-3 ~ 5e-3 | Policy Gradient 通常用较大学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `hidden_dim` | 64 ~ 256 | 隐藏层维度 |
| `num_episodes` | 500 ~ 2000 | 训练回合数 |

---

## 优缺点

### 优点

- ✅ 可以处理连续动作空间
- ✅ 学习的是随机策略，有内在探索能力
- ✅ 不需要学习值函数，更直接
- ✅ 理论保证收敛到局部最优

### 缺点

- ❌ 样本效率低（每个策略更新需要完整轨迹）
- ❌ 方差大，训练不稳定
- ❌ 只能在线学习（episodic）

### 改进方向

- **Actor-Critic**: 加入值函数估计，减少方差
- **PPO**: 使用裁剪目标，稳定训练
- **A3C/A2C**: 异步多线程加速
- **TRPO**: 信任区域方法

---

## 参考文献

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine learning*, 8(3-4), 229-256.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.
