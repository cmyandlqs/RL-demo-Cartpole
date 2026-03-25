# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Conda environment**: Activate the RL environment before running any scripts:
```bash
conda activate RL
```

Python path: `D:\Miniconda3\envs\RL\python.exe`

## Project Overview

强化学习算法实现项目，支持 DQN、Policy Gradient、Actor-Critic、PPO 四种算法在 CartPole 环境下的训练与测试。

## Running the Project

### 训练

```bash
# 使用统一入口脚本（推荐）
python scripts/train.py --algo {dqn|policy_gradient|actor_critic|ppo} --env cartpole

# 或直接运行算法脚本
python -m algorithms.dqn.cartpole.dqn_cartpole
python -m algorithms.policy_gradient.cartpole.policy_gradient_cartpole
python -m algorithms.actor_critic.cartpole.actor_critic_cartpole
python -m algorithms.ppo.cartpole.ppo_cartpole
```

### 测试

```bash
# 基本测试
python scripts/test.py --algo dqn --env cartpole --model outputs/dqn/cartpole/checkpoints/model.pth

# 确定性测试（设置随机种子）
python scripts/test.py --algo ppo --env cartpole --model outputs/ppo/cartpole/checkpoints/model.pth --seed 42

# 无头模式测试
python scripts/test.py --algo actor_critic --env cartpole --model outputs/actor_critic/cartpole/checkpoints/model.pth --no-render
```

## Directory Structure

```
CartPole/
├── algorithms/              # 算法实现（每个目录自包含）
│   ├── dqn/
│   │   └── cartpole/
│   │       └── dqn_cartpole.py
│   ├── policy_gradient/
│   │   └── cartpole/
│   │       ├── policy_gradient_cartpole.py
│   │       └── README.md
│   ├── actor_critic/
│   │   └── cartpole/
│   │       ├── actor_critic_cartpole.py
│   │       └── README.md
│   └── ppo/
│       └── cartpole/
│           ├── ppo_cartpole.py
│           └── README.md
├── outputs/                 # 训练输出（按算法/环境分组）
├── scripts/                 # 统一入口
│   ├── train.py
│   └── test.py
└── docs/
```

## Algorithm Implementations

### DQN (`algorithms/dqn/cartpole/`)

**文件**: `dqn_cartpole.py`

**核心组件**:
- `ReplayBuffer`: 经验回放池，容量 10000
- `QNetwork`: 3层 MLP (256→256→action_dim)
- `DQNAgent`: 包含 q_net 和 target_q_net

**超参数**:
| 参数 | 默认值 |
|-----|-------|
| learning_rate | 2e-3 |
| gamma | 0.9 |
| epsilon_start | 1.0 |
| epsilon_end | 0.01 |
| batch_size | 128 |

### Policy Gradient (`algorithms/policy_gradient/cartpole/`)

**文件**: `policy_gradient_cartpole.py`

**核心组件**:
- `PolicyNetwork`: 3层 MLP (128→128→action_dim) + Softmax
- `REINFORCEAgent`: 策略梯度智能体

**关键方法**:
- `select_action()`: 返回 action，记录 log_prob
- `compute_returns()`: 从后向前计算 G_t，标准化
- `update()`: L = -E[log π(a|s) · G_t]

**超参数**:
| 参数 | 默认值 |
|-----|-------|
| learning_rate | 1e-3 |
| gamma | 0.99 |
| hidden_dim | 128 |

### Actor-Critic (`algorithms/actor_critic/cartpole/`)

**文件**: `actor_critic_cartpole.py`

**核心组件**:
- `ActorNetwork`: 策略网络，输出动作概率
- `CriticNetwork`: 值函数网络，输出 V(s)
- `ActorCriticAgent`: 双网络智能体

**关键方法**:
- `select_action()`: 返回 action, log_prob, value
- `compute_returns_and_advantages()`: A = G_t - V(s)，标准化
- `update()`: actor_loss + critic_loss

**超参数**:
| 参数 | 默认值 |
|-----|-------|
| learning_rate | 1e-3 |
| gamma | 0.99 |
| hidden_dim | 128 |

### PPO (`algorithms/ppo/cartpole/`)

**文件**: `ppo_cartpole.py`

**核心组件**:
- `ActorCriticNet`: 共享 backbone 的 AC 网络
- `RolloutBuffer`: 存储一个更新周期的数据
- `PPOAgent`: PPO 智能体

**关键方法**:
- `compute_gae_and_returns()`: 广义优势估计
- `update()`: 多 epoch 更新，PPO 裁剪目标

**超参数**:
| 参数 | 默认值 |
|-----|-------|
| learning_rate | 3e-4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_epsilon | 0.2 |
| num_epochs | 4 |
| n_episodes_per_update | 10 |

## Code Patterns

### 策略网络采样

```python
from torch.distributions import Categorical

probs = policy_network(state_tensor)
dist = Categorical(probs)
action = dist.sample()
log_prob = dist.log_prob(action)
```

### 折算回报计算（从后向前）

```python
returns = []
G_t = 0
for r in reversed(rewards):
    G_t = r + gamma * G_t
    returns.insert(0, G_t)
returns = torch.tensor(returns)
# 标准化
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

### Top-K 模型保存

```python
top_k_models = []
TOP_K = 5

def save_top_k_model(ep, rew):
    top_k_models.append((ep, rew, path))
    top_k_models.sort(key=lambda x: x[1], reverse=True)
    while len(top_k_models) > TOP_K:
        removed = top_k_models.pop()
        os.remove(removed.path)
```

## Adding New Algorithm

1. 在 `algorithms/{algo_name}/{env_name}/` 下创建 `{algo}_{env}.py`
2. 实现包含 `main()` 函数的训练脚本
3. 在 `scripts/train.py` 添加路由函数
4. 在 `scripts/test.py` 添加测试函数
5. 创建 README.md 说明算法原理

## Dependencies

- `torch` - PyTorch for neural networks
- `gymnasium` - RL environment interface
- `numpy` - Numerical operations
- `tqdm` - Progress bars

## Algorithm Documentation

每个算法目录都包含详细的 README.md：
- [Policy Gradient README](algorithms/policy_gradient/cartpole/README.md)
- [Actor-Critic README](algorithms/actor_critic/cartpole/README.md)
- [PPO README](algorithms/ppo/cartpole/README.md)
