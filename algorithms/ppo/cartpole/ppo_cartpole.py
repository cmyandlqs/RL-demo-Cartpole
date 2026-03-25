"""
PPO (Proximal Policy Optimization) on CartPole-v1

PPO 算法实现，使用裁剪目标稳定训练，支持多轮 epoch 更新

运行方式:
    python -m algorithms.ppo.cartpole.ppo_cartpole
    或
    python scripts/train.py --algo ppo --env cartpole
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import collections
from tqdm import tqdm
from torch.distributions import Categorical


# ==========================================
# 模块 1：Actor-Critic 网络（共享 backbone）
# ==========================================
class ActorCriticNet(nn.Module):
    """共享特征提取的 Actor-Critic 网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCriticNet, self).__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor 头：输出动作概率分布
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic 头：输出状态价值
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """前向传播，返回 (action_probs, value)"""
        features = self.shared(x)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, value

    def act(self, state):
        """采样动作，返回 (action, log_prob, value)"""
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.detach(), value.detach()


# ==========================================
# 模块 2：经验回放缓冲区
# ==========================================
class RolloutBuffer:
    """存储一个更新周期的数据"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get(self):
        """返回所有数据作为张量"""
        return (
            torch.tensor(np.array(self.states), dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.long),
            torch.stack(self.log_probs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.stack(self.values).squeeze(),
            torch.tensor(self.dones, dtype=torch.float32)
        )

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []


# ==========================================
# 模块 3：PPO 智能体
# ==========================================
class PPOAgent:
    """PPO 算法智能体"""

    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Actor-Critic 网络
        self.network = ActorCriticNet(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # 经验缓冲区
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, log_prob, value = self.network.act(state_tensor)
        self.temp_data = (log_prob, value)
        return action

    def store_step(self, state, action, reward, done):
        """存储一步数据"""
        log_prob, value = self.temp_data
        self.buffer.add(state, action, log_prob, reward, value, done)

    def compute_gae_and_returns(self, rewards, values, dones):
        """计算 GAE 和回报"""
        advantages = []
        returns = []
        gae = 0
        next_value = 0

        # 从后向前计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, num_epochs=4, batch_size=64):
        """PPO 更新（多轮 epoch）"""
        # 获取缓冲区数据
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()

        # 计算 GAE 和回报
        advantages, returns = self.compute_gae_and_returns(
            rewards.numpy(), values.numpy(), dones.numpy()
        )

        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        batch_size = min(batch_size, len(states))

        for epoch in range(num_epochs):
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # 重新计算当前策略
                probs, new_values = self.network(batch_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()

                # 重要性采样比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # PPO 裁剪损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)

                # 熵正则化
                entropy_loss = -entropy.mean()

                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        # 清空缓冲区
        self.buffer.clear()

        return (
            total_policy_loss / n_updates,
            total_value_loss / n_updates,
            total_entropy / n_updates
        )

    def save(self, path):
        """保存模型"""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """加载模型"""
        self.network.load_state_dict(torch.load(path))


# ==========================================
# 模块 4：训练主循环
# ==========================================
def main():
    # --- 超参数 ---
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    hidden_dim = 64
    num_episodes = 1000

    # PPO 特有参数
    n_episodes_per_update = 10   # 每次更新收集的 episode 数
    num_epochs = 4                # 每批数据更新的轮数
    batch_size = 64               # mini-batch 大小

    # --- 输出目录 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    save_dir = os.path.join(project_root, "outputs", "ppo", "cartpole", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # --- 初始化环境和智能体 ---
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]   # 4
    action_dim = env.action_space.n              # 2

    agent = PPOAgent(
        state_dim, action_dim, hidden_dim,
        learning_rate, gamma, gae_lambda,
        clip_epsilon, entropy_coef, value_coef
    )

    # --- 训练设置 ---
    top_k_models = []
    TOP_K = 5

    pbar = tqdm(range(num_episodes), desc="PPO-CartPole")
    recent_rewards = []

    def save_top_k_model(ep, rew):
        nonlocal top_k_models
        filename = f'ppo_model_ep{ep+1}_reward{int(rew)}.pth'
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

        # 收集一个 episode 的数据
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.store_step(state, action, reward, terminated or truncated)

            done = terminated or truncated
            episode_reward += reward
            state = next_state

        # 达到指定 episode 数后进行更新
        if (episode + 1) % n_episodes_per_update == 0:
            policy_loss, value_loss, entropy = agent.update(num_epochs, batch_size)

        # 记录奖励
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
        avg_reward = np.mean(recent_rewards)

        # 更新进度条
        pbar.set_postfix({
            'Reward': f'{episode_reward:5.1f}',
            'Avg10': f'{avg_reward:5.1f}'
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
