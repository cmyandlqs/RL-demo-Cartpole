"""
DQN on CartPole-v1
深度 Q 网络 (Deep Q-Network) 实现，解决 CartPole 平衡任务

运行方式:
    python -m algorithms.dqn.cartpole.train
    或从项目根目录:
    python scripts/train.py --algo dqn --env cartpole
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
import os
from tqdm import tqdm


# ==========================================
# 模块 1：经验回放池 (Replay Buffer)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 模块 2：神经网络 (Q-Network)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==========================================
# 模块 3：智能体 (DQN Agent)
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma,
                 epsilon_start, epsilon_end, epsilon_decay):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q_net = QNetwork(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float32)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ==========================================
# 模块 4：训练主循环
# ==========================================
def main():
    # --- 超参数 ---
    learning_rate = 2e-3
    gamma = 0.9
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    buffer_capacity = 10000
    batch_size = 128
    target_update_freq = 10
    num_episodes = 1000

    # --- 输出目录 (相对于项目根目录) ---
    # 获取项目根目录 (向上找到算法目录的父目录)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    save_dir = os.path.join(project_root, "outputs", "dqn", "cartpole", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # --- 初始化环境和组件 ---
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    replay_buffer = ReplayBuffer(buffer_capacity)
    agent = DQNAgent(state_dim, action_dim, learning_rate, gamma,
                     epsilon_start, epsilon_end, epsilon_decay)

    # --- 训练设置 ---
    top_k_models = []
    TOP_K = 5

    pbar = tqdm(range(num_episodes), desc="DQN-CartPole")
    recent_rewards = collections.deque(maxlen=10)
    recent_losses = collections.deque(maxlen=100)

    def save_top_k_model(ep, rew):
        nonlocal top_k_models
        filename = f'dqn_model_ep{ep+1}_reward{int(rew)}.pth'
        save_path = os.path.join(save_dir, filename)
        torch.save(agent.q_net.state_dict(), save_path)

        top_k_models.append((ep + 1, rew, save_path))
        top_k_models.sort(key=lambda x: x[1], reverse=True)

        while len(top_k_models) > TOP_K:
            removed_ep, removed_rew, removed_path = top_k_models.pop()
            if os.path.exists(removed_path):
                os.remove(removed_path)
                pbar.write(f"  删除低分模型: ep{removed_ep}_reward{int(removed_rew)}.pth")

        return top_k_models[0][2]

    # --- 训练循环 ---
    for episode in pbar:
        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, original_reward, terminated, truncated, info = env.step(action)

            # 位置惩罚
            cart_position = next_state[0]
            position_penalty = abs(cart_position) / 2.4
            penalty_coeff = 0.2
            reward = original_reward - penalty_coeff * position_penalty

            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s, 'actions': b_a, 'rewards': b_r,
                    'next_states': b_ns, 'dones': b_d
                }
                loss = agent.update(transition_dict)
                recent_losses.append(loss)

        # Episode 结束
        if episode % target_update_freq == 0:
            agent.sync_target_network()

        agent.update_epsilon()

        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)
        avg_loss = np.mean(recent_losses) if len(recent_losses) > 0 else 0.0

        pbar.set_postfix({
            'Reward': f'{episode_reward:5.1f}',
            'Avg10': f'{avg_reward:5.1f}',
            'Loss': f'{avg_loss:.4f}',
            'Eps': f'{agent.epsilon:.3f}'
        })

        # 保存模型
        if episode_reward >= 500:
            save_top_k_model(episode, episode_reward)
            pbar.write(f" 满分! Episode {episode+1}: {episode_reward}")
        elif episode_reward > 450:
            save_top_k_model(episode, episode_reward)
            pbar.write(f" 高分! Episode {episode+1}: {episode_reward}")

    # 训练完成
    pbar.write("\n" + "="*50)
    pbar.write(f" Top {TOP_K} 模型:")
    for i, (ep, rew, path) in enumerate(top_k_models, 1):
        pbar.write(f"  {i}. Episode {ep}: {rew} → {os.path.basename(path)}")
    pbar.write("="*50)

    env.close()


if __name__ == "__main__":
    main()
