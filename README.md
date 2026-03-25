# Reinforcement Learning Algorithms

[English](./README_EN.md) | 简体中文

强化学习算法实现项目，支持多种算法在不同环境下的训练与测试。

## 支持的算法

| 算法 | 类型 | 状态 |
|-----|------|-----|
| [DQN](algorithms/dqn/cartpole/) | Value-based | ✅ 已实现 |
| Policy Gradient | Policy-based | 🚧 待实现 |
| Actor-Critic | Actor-critic | 🚧 待实现 |
| PPO | Policy-based | 🚧 待实现 |

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
# 训练 DQN on CartPole
python scripts/train.py --algo dqn --env cartpole
```

### 测试

```bash
# 基本测试
python scripts/test.py --algo dqn --env cartpole \
    --model outputs/dqn/cartpole/checkpoints/model.pth

# 确定性测试（设置随机种子）
python scripts/test.py --algo dqn --env cartpole \
    --model outputs/dqn/cartpole/checkpoints/model.pth --seed 42

# 无头模式测试（不显示可视化窗口）
python scripts/test.py --algo dqn --env cartpole \
    --model outputs/dqn/cartpole/checkpoints/model.pth --no-render
```

## 项目结构

```
.
├── algorithms/              # 算法实现
│   ├── dqn/                 # DQN 算法
│   │   ├── cartpole/        # DQN on CartPole
│   │   └── lunarlander/     # DQN on LunarLander
│   ├── policy_gradient/
│   ├── actor_critic/
│   └── ppo/
├── outputs/                 # 训练输出
│   └── dqn/
│       └── cartpole/
│           ├── checkpoints/ # 模型权重
│           └── logs/        # 训练日志
├── scripts/                 # 统一入口
│   ├── train.py            # 训练入口
│   └── test.py             # 测试入口
├── docs/                    # 文档
├── CLAUDE.md                # Claude Code 项目指南
└── README.md                # 本文件
```

## DQN on CartPole

### 环境说明

- **环境**: CartPole-v1
- **状态空间**: 4 维连续向量 `[小车位置, 小车速度, 杆子角度, 杆角速度]`
- **动作空间**: 2 个离散动作 (0 = 向左推, 1 = 向右推)
- **目标**: 尽可能长时间保持杆子平衡（最多 500 步）

### 算法特点

- **经验回放** (Experience Replay): 打破时间相关性
- **目标网络** (Target Network): 稳定训练
- **ε-greedy 探索**: 平衡探索与利用
- **位置惩罚**: 防止小车撞墙

### 超参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `learning_rate` | 2e-3 | Adam 学习率 |
| `gamma` | 0.9 | 折扣因子 |
| `epsilon_start` | 1.0 | 初始探索率 |
| `epsilon_end` | 0.01 | 最小探索率 |
| `epsilon_decay` | 0.995 | 探索率衰减 |
| `buffer_capacity` | 10000 | 经验池容量 |
| `batch_size` | 128 | 批量大小 |
| `target_update_freq` | 10 | 目标网络同步频率 |

### 训练结果

训练会自动保存 Top-5 高分模型到 `outputs/dqn/cartpole/checkpoints/`。

## 支持的环境

| 环境 | 状态空间 | 动作空间 |
|-----|---------|---------|
| CartPole-v1 | 4 (连续) | 2 (离散) |
| LunarLander-v2 | 8 (连续) | 4 (离散) |

## 添加新算法

1. 在 `algorithms/{algo_name}/{env_name}/` 下创建 `{algo}_{env}.py`
2. 实现包含 `main()` 函数的训练脚本
3. 在 `scripts/train.py` 添加路由函数
4. 在 `scripts/test.py` 添加测试函数

## 参考文档

- [CLAUDE.md](./CLAUDE.md) - 项目开发指南
- [docs/Gemini RL环境说明文档.md](./docs/Gemini%20RL%E7%8E%AF%E5%A2%83%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md) - CartPole 环境与 DQN 原理

## 许可证

MIT License
