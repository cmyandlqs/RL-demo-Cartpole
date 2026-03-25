# Reinforcement Learning Algorithms

English | [简体中文](./README.md)

A collection of reinforcement learning algorithm implementations, supporting training and testing across multiple algorithms and environments.

## Supported Algorithms

| Algorithm | Type | Status |
|-----------|------|--------|
| [DQN](algorithms/dqn/cartpole/) | Value-based | ✅ Implemented |
| Policy Gradient | Policy-based | 🚧 TODO |
| Actor-Critic | Actor-critic | 🚧 TODO |
| PPO | Policy-based | 🚧 TODO |

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
# Train DQN on CartPole
python scripts/train.py --algo dqn --env cartpole
```

### Testing

```bash
# Basic testing
python scripts/test.py --algo dqn --env cartpole \
    --model outputs/dqn/cartpole/checkpoints/model.pth

# Deterministic testing (with random seed)
python scripts/test.py --algo dqn --env cartpole \
    --model outputs/dqn/cartpole/checkpoints/model.pth --seed 42

# Headless mode (no visualization window)
python scripts/test.py --algo dqn --env cartpole \
    --model outputs/dqn/cartpole/checkpoints/model.pth --no-render
```

## Project Structure

```
.
├── algorithms/              # Algorithm implementations
│   ├── dqn/                 # DQN algorithm
│   │   ├── cartpole/        # DQN on CartPole
│   │   └── lunarlander/     # DQN on LunarLander
│   ├── policy_gradient/
│   ├── actor_critic/
│   └── ppo/
├── outputs/                 # Training outputs
│   └── dqn/
│       └── cartpole/
│           ├── checkpoints/ # Model weights
│           └── logs/        # Training logs
├── scripts/                 # Unified entry points
│   ├── train.py            # Training entry
│   └── test.py             # Testing entry
├── docs/                    # Documentation
├── CLAUDE.md                # Project guide for Claude Code
└── README.md                # This file
```

## DQN on CartPole

### Environment Details

- **Environment**: CartPole-v1
- **State Space**: 4-dimensional continuous vector `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`
- **Action Space**: 2 discrete actions (0 = push left, 1 = push right)
- **Goal**: Keep the pole balanced as long as possible (max 500 steps)

### Algorithm Features

- **Experience Replay**: Break temporal correlations
- **Target Network**: Stabilize training
- **ε-greedy Exploration**: Balance exploration vs exploitation
- **Position Penalty**: Prevent cart from hitting boundaries

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2e-3 | Adam learning rate |
| `gamma` | 0.9 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Per-episode decay |
| `buffer_capacity` | 10000 | Replay buffer capacity |
| `batch_size` | 128 | Batch size |
| `target_update_freq` | 10 | Target network sync frequency |

### Training Results

Training automatically saves Top-5 highest reward models to `outputs/dqn/cartpole/checkpoints/`.

## Supported Environments

| Environment | State Space | Action Space |
|-------------|-------------|--------------|
| CartPole-v1 | 4 (continuous) | 2 (discrete) |
| LunarLander-v2 | 8 (continuous) | 4 (discrete) |

## Adding New Algorithms

1. Create `{algo}_{env}.py` in `algorithms/{algo_name}/{env_name}/`
2. Implement training script with `main()` function
3. Add routing function in `scripts/train.py`
4. Add testing function in `scripts/test.py`

## References

- [CLAUDE.md](./CLAUDE.md) - Project development guide
- [docs/Gemini RL环境说明文档.md](./docs/Gemini%20RL%E7%8E%AF%E5%A2%83%E8%AF%B4%E6%98%8E%E6%96%87%E6%A1%A3.md) - CartPole environment & DQN principles (Chinese)

## License

MIT License
