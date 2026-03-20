# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 代码运行环境
在conda环境中的RL环境中，'conda activate RL' 激活RL环境
## Project Overview

This is a Deep Q-Network (DQN) reinforcement learning implementation for solving the CartPole-v1 environment from Gymnasium. The project demonstrates the classic DQN algorithm with experience replay, target networks, and epsilon-greedy exploration.

### Environment Details

- **Environment**: CartPole-v1
- **State Space**: 4-dimensional continuous vector `[cart_position, cart_velocity, pole_angle, pole_velocity]`
- **Action Space**: Discrete 2 actions (0 = push left, 1 = push right)
- **Goal**: Keep the pole balanced upright as long as possible (max 500 steps/episode)

## Running the Project

```bash
# Test the CartPole environment with random actions
python test_env.py

# Train the DQN agent
python dqn_cartpole.py
```

## Architecture

The codebase follows a modular structure with 4 core components:

### 1. ReplayBuffer (`dqn_cartpole.py:14-34`)
- Uses `collections.deque` with fixed capacity for automatic eviction of oldest experiences
- Stores transitions as tuples: `(state, action, reward, next_state, done)`
- `push()`: Add new experience
- `sample()`: Random sampling for training breaks temporal correlation

### 2. QNetwork (`dqn_cartpole.py:40-51`)
- Simple 3-layer MLP: `input -> 128 -> 128 -> output`
- Input: state (4-dim), Output: Q-values for each action (2-dim)
- ReLU activation for hidden layers, no activation on output

### 3. DQNAgent (`dqn_cartpole.py:57-124`)
- **Double network architecture**: `q_net` (online/eval) and `target_q_net` (stable target)
- `select_action()`: Epsilon-greedy policy (explore with prob ε, exploit otherwise)
- `update()`: Bellman equation loss with MSE
- `sync_target_network()`: Copy q_net weights to target_q_net
- `update_epsilon()`: Decay exploration rate

### 4. Training Loop (`dqn_cartpole.py:130-199`)
- Episode-based training (default: 300 episodes)
- Target network sync every 10 episodes
- Training starts when buffer size > batch_size

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2e-3 | Adam optimizer learning rate |
| `gamma` | 0.98 | Discount factor for future rewards |
| `epsilon_start` | 1.0 | Initial exploration rate (100% random) |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Exploration decay per update |
| `buffer_capacity` | 10000 | Replay buffer size |
| `batch_size` | 64 | Training batch size |
| `target_update_freq` | 10 | Episodes between target network sync |

## Dependencies

- `torch` - PyTorch for neural networks
- `gymnasium` - RL environment interface
- `numpy` - Numerical operations

Note: Use Conda for environment management (configured in `.vscode/settings.json`).

## File Conventions

- **Chinese comments**: Code contains extensive Chinese comments explaining the algorithm
- **公式文档.md**: Chinese documentation with pseudocode and algorithm explanation
- **codding_learn.ipynb**: Experimental notebook for understanding replay buffer mechanics
