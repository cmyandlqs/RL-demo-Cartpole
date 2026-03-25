# Reinforcement Learning Algorithms

[English](./README_EN.md) | 简体中文

强化学习算法实现项目，支持多种算法在不同环境下的训练与测试。

## 支持的算法

| 算法 | 类型 | 状态 | 文档 |
|-----|------|-----|------|
| [DQN](algorithms/dqn/cartpole/) | Value-based | ✅ 已实现 | [README](algorithms/dqn/cartpole/) |
| [Policy Gradient](algorithms/policy_gradient/cartpole/) | Policy-based | ✅ 已实现 | [README](algorithms/policy_gradient/cartpole/README.md) |
| [Actor-Critic](algorithms/actor_critic/cartpole/) | Actor-critic | ✅ 已实现 | [README](algorithms/actor_critic/cartpole/README.md) |
| [PPO](algorithms/ppo/cartpole/) | Policy-based | ✅ 已实现 | [README](algorithms/ppo/cartpole/README.md) |

### 算法对比

| 特性 | DQN | Policy Gradient | Actor-Critic | PPO |
|-----|-----|----------------|--------------|-----|
| 学习方式 | Q 值函数 | 策略梯度 | 策略+价值 | 策略+价值 |
| 数据效率 | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| 训练稳定性 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 连续动作 | ❌ | ✅ | ✅ | ✅ |
| 实现难度 | 中 | 简单 | 中 | 中 |

## 环境要求

```bash
# 创建 conda 环境
conda create -n RL python=3.10
conda activate RL

# 安装依赖
pip install torch gymnasium numpy tqdm
```

## 快速开始

### 训练

```bash
# DQN
python scripts/train.py --algo dqn --env cartpole

# Policy Gradient
python scripts/train.py --algo policy_gradient --env cartpole

# Actor-Critic
python scripts/train.py --algo actor_critic --env cartpole

# PPO
python scripts/train.py --algo ppo --env cartpole
```

### 测试

```bash
# 基本测试
python scripts/test.py --algo dqn --env cartpole \
    --model outputs/dqn/cartpole/checkpoints/model.pth

# 确定性测试（设置随机种子）
python scripts/test.py --algo ppo --env cartpole \
    --model outputs/ppo/cartpole/checkpoints/model.pth --seed 42

# 无头模式测试
python scripts/test.py --algo actor_critic --env cartpole \
    --model outputs/actor_critic/cartpole/checkpoints/model.pth --no-render
```

## 项目结构

```
.
├── algorithms/              # 算法实现
│   ├── dqn/                 # DQN 算法
│   ├── policy_gradient/     # Policy Gradient 算法
│   ├── actor_critic/        # Actor-Critic 算法
│   └── ppo/                 # PPO 算法
├── outputs/                 # 训练输出
│   ├── dqn/
│   ├── policy_gradient/
│   ├── actor_critic/
│   └── ppo/
├── scripts/                 # 统一入口
│   ├── train.py            # 训练入口
│   └── test.py             # 测试入口
├── docs/                    # 文档
├── CLAUDE.md                # 项目指南
└── README.md                # 本文件
```

## CartPole-v1 环境

- **状态空间**: 4 维 `[小车位置, 小车速度, 杆子角度, 杆角速度]`
- **动作空间**: 2 个离散动作 (0 = 向左, 1 = 向右)
- **目标**: 尽可能长时间保持杆子平衡（最多 500 步）

## 算法详解

### DQN (Deep Q-Network)

- **核心**: 学习 Q(s,a) 值函数
- **技术**: 经验回放 + 目标网络
- **适用**: 离散动作空间

详细说明: [algorithms/dqn/cartpole/](algorithms/dqn/cartpole/)

### Policy Gradient (REINFORCE)

- **核心**: 直接优化策略 π(a|s)
- **公式**: ∇J = E[log π(a|s) · G_t]
- **特点**: 高方差，但理论保证收敛

详细说明: [algorithms/policy_gradient/cartpole/README.md](algorithms/policy_gradient/cartpole/README.md)

### Actor-Critic

- **核心**: Actor 优化策略，Critic 估计价值
- **优势**: 使用 A(s,a) = G_t - V(s) 减少方差
- **特点**: 比纯 Policy Gradient 更高效

详细说明: [algorithms/actor_critic/cartpole/README.md](algorithms/actor_critic/cartpole/README.md)

### PPO (Proximal Policy Optimization)

- **核心**: 裁剪策略更新幅度
- **公式**: L = -E[min(r·A, clip(r)·A)]
- **特点**: 稳定训练，多次利用数据

详细说明: [algorithms/ppo/cartpole/README.md](algorithms/ppo/cartpole/README.md)

## 超参数速查

| 算法 | Learning Rate | Gamma | 特殊参数 |
|-----|--------------|-------|---------|
| DQN | 2e-3 | 0.9 | ε: 1.0→0.01 |
| PG | 1e-3 | 0.99 | - |
| AC | 1e-3 | 0.99 | - |
| PPO | 3e-4 | 0.99 | clip=0.2, λ=0.95 |

## 参考文档

- [CLAUDE.md](./CLAUDE.md) - 项目开发指南
- [docs/Gemini RL环境说明文档.md](./docs/Gemini%20RL%E7%8E%AF%E5%A2%83%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md) - CartPole 环境与 DQN 原理

## 许可证

MIT License
