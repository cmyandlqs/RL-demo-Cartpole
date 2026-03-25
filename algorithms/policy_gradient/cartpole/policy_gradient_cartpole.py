"""
Policy Gradient (REINFORCE) on CartPole-v1

策略梯度算法实现，通过直接优化策略函数来学习最优策略

运行方式:
    python -m algorithms.policy_gradient.cartpole.policy_gradient_cartpole
    或
    python scripts/train.py --algo policy_gradient --env cartpole

参考文献:
    Williams, R. J. (1992). Simple statistical gradient-following algorithms
    for connectionist reinforcement learning.
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from torch.distributions import Categorical


# ==========================================
# 模块 1：策略网络 (Policy Network)
# ==========================================
class PolicyNetwork(nn.Module):
    """策略网络：输入状态，输出动作概率分布"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Softmax 确保输出是概率分布（和为1）
        return F.softmax(self.fc3(x), dim=-1)

    def act(self, state):
        """根据当前状态选择动作，返回 (action, log_prob)"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.forward(state_tensor)

        # 创建分类分布并采样
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


# ==========================================
# 模块 2：REINFORCE 智能体
# ==========================================
class REINFORCEAgent:
    """REINFORCE 算法智能体"""

    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 learning_rate=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # 记录一个 episode 的数据
        self.log_probs = []      # 每步的 log π(a|s)
        self.rewards = []        # 每步的 reward

    def select_action(self, state):
        """选择动作，并记录 log_prob"""
        action, log_prob = self.policy_net.act(state)
        self.log_probs.append(log_prob)
        return action

    def store_reward(self, reward):
        """记录奖励"""
        self.rewards.append(reward)

    def compute_returns(self):
        """从后向前计算折算回报 G_t"""
        returns = []
        G_t = 0
        # 从最后一个 reward 往前计算
        for r in reversed(self.rewards):
            G_t = r + self.gamma * G_t
            returns.insert(0, G_t)

        returns = torch.tensor(returns, dtype=torch.float32)

        # 标准化回报（有助于训练稳定）
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns

    def update(self):
        """根据采样的轨迹更新策略"""
        returns = self.compute_returns()

        # 计算损失: L = -E[log π(a|s) * G_t]
        loss = []
        for log_prob, G_t in zip(self.log_probs, returns):
            # 负号因为我们要最大化，但 PyTorch 是最小化
            loss.append(-log_prob * G_t)

        # 将所有时间步的损失求和
        loss = torch.stack(loss).sum()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空记录，准备下一个 episode
        self.log_probs = []
        self.rewards = []

        return loss.item()

    def save(self, path):
        """保存模型"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """加载模型"""
        self.policy_net.load_state_dict(torch.load(path))


# ==========================================
# 模块 3：训练主循环
# ==========================================
def main():
    # --- 超参数 ---
    learning_rate = 1e-3
    gamma = 0.99
    hidden_dim = 128
    num_episodes = 1000

    # --- 输出目录 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    save_dir = os.path.join(project_root, "outputs", "policy_gradient", "cartpole", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # --- 初始化环境和智能体 ---
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n              # 2

    agent = REINFORCEAgent(state_dim, action_dim, hidden_dim,
                          learning_rate, gamma)

    # --- 训练设置 ---
    top_k_models = []
    TOP_K = 5

    pbar = tqdm(range(num_episodes), desc="PG-CartPole")
    recent_rewards = []

    def save_top_k_model(ep, rew):
        nonlocal top_k_models
        filename = f'pg_model_ep{ep+1}_reward{int(rew)}.pth'
        save_path = os.path.join(save_dir, filename)
        agent.save(save_path)

        top_k_models.append((ep + 1, rew, save_path))
        top_k_models.sort(key=lambda x: x[1], reverse=True)

        while len(top_k_models) > TOP_K:
            removed_ep, removed_rew, removed_path = top_k_models.pop()
            if os.path.exists(removed_path):
                os.remove(removed_path)

        return top_k_models[0][2]

    # --- 训练循环 ---
    for episode in pbar:
        state, info = env.reset()
        done = False
        episode_reward = 0

        # 采样一条完整轨迹
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.store_reward(reward)

            done = terminated or truncated
            episode_reward += reward
            state = next_state

        # Episode 结束，计算回报并更新策略
        loss = agent.update()

        # 记录奖励
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
        avg_reward = np.mean(recent_rewards)

        # 更新进度条
        pbar.set_postfix({
            'Reward': f'{episode_reward:5.1f}',
            'Avg10': f'{avg_reward:5.1f}',
            'Loss': f'{loss:.4f}'
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
