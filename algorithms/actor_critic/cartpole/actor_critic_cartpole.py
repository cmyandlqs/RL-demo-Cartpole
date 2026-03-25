"""
Actor-Critic on CartPole-v1

Actor-Critic 算法实现，结合策略梯度和值函数，减少方差提高样本效率

运行方式:
    python -m algorithms.actor_critic.cartpole.actor_critic_cartpole
    或
    python scripts/train.py --algo actor_critic --env cartpole
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
# 模块 1：Actor 网络（策略网络）
# ==========================================
class ActorNetwork(nn.Module):
    """策略网络：输入状态，输出动作概率分布"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


# ==========================================
# 模块 2：Critic 网络（值函数网络）
# ==========================================
class CriticNetwork(nn.Module):
    """值函数网络：输入状态，输出状态价值 V(s)"""

    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出单个标量

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==========================================
# 模块 3：Actor-Critic 智能体
# ==========================================
class ActorCriticAgent:
    """Actor-Critic 算法智能体"""

    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 learning_rate=1e-3, gamma=0.99):
        self.gamma = gamma

        # Actor 和 Critic 网络
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, hidden_dim)

        # 共享优化器（也可以分别设置）
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=learning_rate
        )

        # 记录一个 episode 的数据
        self.log_probs = []      # log π(a|s)
        self.values = []         # V(s)
        self.rewards = []        # rewards

    def select_action(self, state):
        """选择动作，返回 (action, log_prob, value)"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Actor 输出动作概率
        probs = self.actor(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic 输出状态价值
        value = self.critic(state_tensor)

        # 存储用于更新
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action.item()

    def store_reward(self, reward):
        """记录奖励"""
        self.rewards.append(reward)

    def compute_returns_and_advantages(self):
        """计算回报和优势函数"""
        returns = []
        advantages = []

        G_t = 0
        # 从后向前计算
        for r, v in zip(reversed(self.rewards), reversed(self.values)):
            G_t = r + self.gamma * G_t
            returns.insert(0, G_t)
            # 优势 = 回报 - 价值估计
            advantages.insert(0, G_t - v.item())

        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # 标准化优势（训练更稳定）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self):
        """更新 Actor 和 Critic"""
        returns, advantages = self.compute_returns_and_advantages()

        # 将存储的 log_probs 和 values 转为张量
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()

        # === Actor 损失 ===
        # L_actor = -E[log π(a|s) * A(s,a)]
        # 优势不参与梯度计算（detach）
        actor_loss = -(log_probs * advantages.detach()).mean()

        # === Critic 损失 ===
        # L_critic = MSE(V(s), G_t)
        critic_loss = F.mse_loss(values, returns)

        # === 总损失 ===
        loss = actor_loss + critic_loss

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 清空记录
        self.log_probs = []
        self.values = []
        self.rewards = []

        return loss.item(), actor_loss.item(), critic_loss.item()

    def save(self, path):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


# ==========================================
# 模块 4：训练主循环
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
    save_dir = os.path.join(project_root, "outputs", "actor_critic", "cartpole", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # --- 初始化环境和智能体 ---
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n              # 2

    agent = ActorCriticAgent(state_dim, action_dim, hidden_dim,
                             learning_rate, gamma)

    # --- 训练设置 ---
    top_k_models = []
    TOP_K = 5

    pbar = tqdm(range(num_episodes), desc="AC-CartPole")
    recent_rewards = []

    def save_top_k_model(ep, rew):
        nonlocal top_k_models
        filename = f'ac_model_ep{ep+1}_reward{int(rew)}.pth'
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

        # Episode 结束，更新 Actor 和 Critic
        loss, actor_loss, critic_loss = agent.update()

        # 记录奖励
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
        avg_reward = np.mean(recent_rewards)

        # 更新进度条
        pbar.set_postfix({
            'Reward': f'{episode_reward:5.1f}',
            'Avg10': f'{avg_reward:5.1f}',
            'ALoss': f'{actor_loss:.4f}',
            'CLoss': f'{critic_loss:.4f}'
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
