# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Conda environment**: Activate the RL environment before running any scripts:
```bash
conda activate RL
```

Python path: `D:\Miniconda3\envs\RL\python.exe`

## Project Overview

强化学习算法实现项目，支持多种算法和环境的组合训练与测试。

**项目结构**:
- 每个算法目录完全独立，便于复制和分享
- 统一的训练/测试入口，通过命令行参数指定算法和环境
- 输出按 `outputs/{algo}/{env}/` 结构组织

## Directory Structure

```
CartPole/
├── algorithms/                    # 算法实现（每个目录自包含）
│   ├── dqn/
│   │   └── cartpole/
│   │       └── train.py          # DQN on CartPole
│   ├── policy_gradient/
│   ├── actor_critic/
│   └── ppo/
├── outputs/                       # 训练输出
│   └── dqn/
│       └── cartpole/
│           ├── checkpoints/      # 模型权重
│           └── logs/             # 训练日志
├── scripts/                       # 统一入口
│   ├── train.py                  # 训练入口
│   └── test.py                   # 测试入口
└── docs/                         # 文档
```

## Running the Project

### 训练

```bash
# 使用统一入口脚本（推荐）
python scripts/train.py --algo dqn --env cartpole

# 或直接运行算法脚本
python -m algorithms.dqn.cartpole.train
```

### 测试

```bash
# 基本测试
python scripts/test.py --algo dqn --env cartpole --model outputs/dqn/cartpole/checkpoints/model.pth

# 确定性测试（设置随机种子）
python scripts/test.py --algo dqn --env cartpole --model outputs/dqn/cartpole/checkpoints/model.pth --seed 42

# 无头模式测试
python scripts/test.py --algo dqn --env cartpole --model outputs/dqn/cartpole/checkpoints/model.pth --no-render
```

## Supported Algorithms & Environments

| Algorithm | Environments |
|-----------|--------------|
| DQN | cartpole, lunarlander |
| Policy Gradient | cartpole |
| Actor-Critic | cartpole |
| PPO | cartpole, lunarlander |

## Adding New Algorithm

1. 在 `algorithms/{algo_name}/{env_name}/` 下创建 `train.py`
2. 确保训练代码包含 `main()` 函数作为入口
3. 在 `scripts/train.py` 中添加对应的路由函数
4. 在 `scripts/test.py` 中添加对应的测试函数

## CartPole Environment Details

- **Environment**: CartPole-v1
- **State Space**: 4-dimensional `[cart_position, cart_velocity, pole_angle, pole_velocity]`
- **Action Space**: Discrete 2 (0 = left, 1 = right)
- **Goal**: Balance pole for 500 steps

## DQN Implementation Notes

当前 DQN 实现 (`algorithms/dqn/cartpole/train.py`) 特点：

- **ReplayBuffer**: `collections.deque`，容量 10000
- **QNetwork**: 3-layer MLP `256→256→action_dim`
- **双网络**: `q_net` (online) + `target_q_net`
- **ε-greedy**: 从 1.0 衰减到 0.01
- **位置惩罚**: `penalty_coeff=0.2`，防止小车撞墙
- **Top-K 保存**: 自动保留 reward 最高的 5 个模型

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2e-3 | Adam learning rate |
| `gamma` | 0.9 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration |
| `epsilon_end` | 0.01 | Minimum exploration |
| `epsilon_decay` | 0.995 | Per-episode decay |
| `buffer_capacity` | 10000 | Replay buffer size |
| `batch_size` | 128 | Training batch size |
| `target_update_freq` | 10 | Episodes between sync |
| `num_episodes` | 1000 | Total episodes |

## Dependencies

- `torch` - PyTorch for neural networks
- `gymnasium` - RL environment interface
- `numpy` - Numerical operations
- `tqdm` - Progress bars

## Code Patterns

- **State to Tensor**: `torch.tensor(state, dtype=torch.float32).unsqueeze(0)`
- **Action Selection**: `q_values.argmax().item()`
- **Target Q**: `rewards + gamma * max_next_q * (1 - dones)`
- **Progress Tracking**: `collections.deque(maxlen=N)` for rolling averages

## Legacy Files

以下文件为旧版本，保留用于参考：
- `dqn_cartpole.py` - 旧版 DQN 训练脚本
- `test_trained_model.py` - 旧版测试脚本
- `test_env.py` - 环境测试脚本

新代码请使用 `scripts/` 下的统一入口。
