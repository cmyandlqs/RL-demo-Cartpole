# Reinforcement Learning Algorithms

English | [简体中文](./README.md)

A collection of reinforcement learning algorithm implementations, supporting training and testing across multiple algorithms and environments.

## Supported Algorithms

| Algorithm | Type | Status | Docs |
|-----------|------|--------|------|
| [DQN](algorithms/dqn/cartpole/) | Value-based | ✅ Implemented | [README](algorithms/dqn/cartpole/) |
| [Policy Gradient](algorithms/policy_gradient/cartpole/) | Policy-based | ✅ Implemented | [README](algorithms/policy_gradient/cartpole/README.md) |
| [Actor-Critic](algorithms/actor_critic/cartpole/) | Actor-critic | ✅ Implemented | [README](algorithms/actor_critic/cartpole/README.md) |
| [PPO](algorithms/ppo/cartpole/) | Policy-based | ✅ Implemented | [README](algorithms/ppo/cartpole/README.md) |

### Algorithm Comparison

| Feature | DQN | Policy Gradient | Actor-Critic | PPO |
|---------|-----|----------------|--------------|-----|
| Learning | Q-function | Policy gradient | Policy+Value | Policy+Value |
| Data Efficiency | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| Stability | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Continuous Actions | ❌ | ✅ | ✅ | ✅ |
| Difficulty | Medium | Simple | Medium | Medium |

## Requirements

```bash
# Create conda environment
conda create -n RL python=3.10
conda activate RL

# Install dependencies
pip install torch gymnasium numpy tqdm
```

## Quick Start

### Training

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

### Testing

```bash
# Basic testing
python scripts/test.py --algo dqn --env cartpole \
    --model outputs/dqn/cartpole/checkpoints/model.pth

# Deterministic testing
python scripts/test.py --algo ppo --env cartpole \
    --model outputs/ppo/cartpole/checkpoints/model.pth --seed 42

# Headless mode
python scripts/test.py --algo actor_critic --env cartpole \
    --model outputs/actor_critic/cartpole/checkpoints/model.pth --no-render
```

## Project Structure

```
.
├── algorithms/              # Algorithm implementations
│   ├── dqn/                 # DQN
│   ├── policy_gradient/     # Policy Gradient
│   ├── actor_critic/        # Actor-Critic
│   └── ppo/                 # PPO
├── outputs/                 # Training outputs
├── scripts/                 # Entry points
│   ├── train.py
│   └── test.py
├── docs/                    # Documentation
├── CLAUDE.md                # Project guide
└── README.md                # This file
```

## CartPole-v1 Environment

- **State Space**: 4D `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`
- **Action Space**: 2 discrete actions (0 = push left, 1 = push right)
- **Goal**: Keep the pole balanced as long as possible (max 500 steps)

## Algorithm Details

### DQN (Deep Q-Network)

- **Core**: Learn Q(s,a) value function
- **Techniques**: Experience Replay + Target Network
- **Use Case**: Discrete action spaces

Details: [algorithms/dqn/cartpole/](algorithms/dqn/cartpole/)

### Policy Gradient (REINFORCE)

- **Core**: Directly optimize policy π(a|s)
- **Formula**: ∇J = E[log π(a|s) · G_t]
- **Feature**: High variance, but guaranteed convergence

Details: [algorithms/policy_gradient/cartpole/README.md](algorithms/policy_gradient/cartpole/README.md)

### Actor-Critic

- **Core**: Actor optimizes policy, Critic estimates value
- **Advantage**: Uses A(s,a) = G_t - V(s) to reduce variance
- **Feature**: More efficient than pure Policy Gradient

Details: [algorithms/actor_critic/cartpole/README.md](algorithms/actor_critic/cartpole/README.md)

### PPO (Proximal Policy Optimization)

- **Core**: Clip policy updates to prevent large changes
- **Formula**: L = -E[min(r·A, clip(r)·A)]
- **Feature**: Stable training, multiple data reuse

Details: [algorithms/ppo/cartpole/README.md](algorithms/ppo/cartpole/README.md)

## Hyperparameter Quick Reference

| Algorithm | Learning Rate | Gamma | Special Params |
|-----------|--------------|-------|----------------|
| DQN | 2e-3 | 0.9 | ε: 1.0→0.01 |
| PG | 1e-3 | 0.99 | - |
| AC | 1e-3 | 0.99 | - |
| PPO | 3e-4 | 0.99 | clip=0.2, λ=0.95 |

## References

- [CLAUDE.md](./CLAUDE.md) - Project development guide
- [docs/Gemini RL环境说明文档.md](./docs/Gemini%20RL%E7%8E%AF%E5%A2%83%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md) - CartPole environment & DQN principles (Chinese)

## License

MIT License
