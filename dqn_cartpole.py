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
# 作用：充当智能体的“记忆库”，存放历史的游玩数据，打破时间相关性
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        # 使用双端队列 deque，当存满 capacity 时，会自动把最老的数据挤掉
        self.buffer = collections.deque(maxlen=capacity) 

    def push(self, state, action, reward, next_state, done):
        # 将这一步的经历打包成元组，存入记忆库
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size):
        # 随机从记忆库中抓取 batch_size 条数据
        transitions = random.sample(self.buffer, batch_size)
        # 将数据解包并按列组合 (比如把所有的 state 放在一起，所有的 action 放在一起)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def __len__(self):
        # 返回当前记忆库里的数据量
        return len(self.buffer)

# ==========================================
# 模块 2：神经网络 (Q-Network)
# 作用：智能体的大脑，输入状态，输出每个动作的 Q 值 (预期得分)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 一个简单的三层全连接神经网络 (MLP)
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) # 输出层不需要激活函数，直接输出 Q 值

# ==========================================
# 模块 3：智能体 (DQN Agent)
# 作用：封装动作选择 (Actor) 和 网络更新 (Learner) 的逻辑
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay):
        self.action_dim = action_dim
        self.gamma = gamma                # 折扣因子 (对未来奖励的看重程度)
        self.epsilon = epsilon_start      # 初始探索概率 (瞎走的概率)
        self.epsilon_end = epsilon_end    # 最低探索概率
        self.epsilon_decay = epsilon_decay# 探索概率的衰减率
        
        # 核心：实例化 评估网络(online) 和 目标网络(target)
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q_net = QNetwork(state_dim, action_dim)
        # 初始化时，让目标网络的权重和评估网络完全一样
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # 优化器，只更新 q_net 的参数
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def select_action(self, state):
        # epsilon-贪婪策略：一定概率随机探索，一定概率利用现有知识
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim) # 瞎选 (返回 0 或 1)
        else:
            # 把 numpy 数组转成 PyTorch 张量
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # 拿到 Q 预测值，不需要计算梯度
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            # 选 Q 值最大的那个动作的索引
            return q_values.argmax().item()

    def update(self, transition_dict):
        # 1. 提取并转换批量数据为 PyTorch 的 Tensor
        states = torch.tensor(transition_dict['states'], dtype=torch.float32)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1) # 变成列向量
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1)

        # 2. 计算【当前网络的预测 Q 值】
        # q_net(states) 输出所有动作的 Q 值，.gather(1, actions) 是把我们实际采样的那个动作的 Q 值挑出来
        q_values = self.q_net(states).gather(1, actions)

        # 3. 计算【目标 Q 值 (Ground Truth)】
        # 下一个状态的最大 Q 值是由 target_q_net 算出来的！(切断梯度)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 核心贝尔曼方程：如果 done=1，(1-done)=0，只拿即时奖励；否则加上未来的预期奖励
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 4. 计算 MSE 误差并反向传播
        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad() # 清空旧梯度
        loss.backward()            # 反向传播算梯度
        self.optimizer.step()      # 梯度下降更新权重

        return loss.item()  # 返回 loss 值用于显示

    def sync_target_network(self):
        # 同步函数：把 q_net 的脑子完整地复制给 target_q_net
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
    def update_epsilon(self):
        # 每次学习完，慢慢减少瞎走的概率，让模型越来越自信
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

# ==========================================
# 模块 4：训练主循环 (Main Loop)
# 作用：让智能体在环境里打怪升级
# ==========================================
def main():
    # --- 超参数设置 ---
    learning_rate = 2e-3      # 学习率
    gamma = 0.9              # 折扣因子
    epsilon_start = 1.0       # 一开始 100% 瞎走探索
    epsilon_end = 0.01        # 最后保留 1% 的概率探索
    epsilon_decay = 0.995     # 探索率衰减系数
    buffer_capacity = 10000   # 经验池大小
    batch_size = 64           # 每次从经验池抓几条数据学习
    target_update_freq = 10   # 每隔 10 个 Episode 同步一次目标网络
    num_episodes = 500        # 总共玩 500 局游戏（增加训练量）

    # --- 初始化环境和组件 ---
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0] # 4 维
    action_dim = env.action_space.n            # 2 维 (0向左, 1向右)
    
    replay_buffer = ReplayBuffer(buffer_capacity)
    agent = DQNAgent(state_dim, action_dim, learning_rate, gamma, 
                     epsilon_start, epsilon_end, epsilon_decay)

    # --- 开始打游戏 ---
    # 创建保存权重的目录
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # 使用 tqdm 显示进度条
    pbar = tqdm(range(num_episodes), desc="Training")
    recent_rewards = collections.deque(maxlen=10)  # 记录最近10局的平均奖励
    recent_losses = collections.deque(maxlen=100)   # 记录最近100次更新的 loss

    for episode in pbar:
        state, info = env.reset() # 每一局开始，重置环境
        done = False
        episode_reward = 0        # 记录这一局拿了多少分

        while not done:
            # 1. 智能体根据当前状态选择动作
            action = agent.select_action(state)
            
            # 2. 环境执行动作，返回结果
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 3. 把经历存进经验池
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            # 4. 如果经验池里的数据足够多，就开始学习！
            if len(replay_buffer) > batch_size:
                # 采样数据
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                # 打包成字典传给智能体
                transition_dict = {
                    'states': b_s, 'actions': b_a, 'rewards': b_r,
                    'next_states': b_ns, 'dones': b_d
                }
                # 智能体通过反向传播自我进化，并返回 loss
                loss = agent.update(transition_dict)
                recent_losses.append(loss)

        # 一局结束，判断是否需要同步目标网络
        if episode % target_update_freq == 0:
            agent.sync_target_network()

        # 记录最近的奖励，用于计算平均值
        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)

        # 每 episode 结束后降低探索率（而不是每 step）
        agent.update_epsilon()

        # 计算平均 loss（如果没有训练过则为 0）
        avg_loss = np.mean(recent_losses) if len(recent_losses) > 0 else 0.0

        # 更新进度条显示
        pbar.set_postfix({
            'Reward': f'{episode_reward:5.1f}',
            'Avg10': f'{avg_reward:5.1f}',
            'Loss': f'{avg_loss:.4f}',
            'Epsilon': f'{agent.epsilon:.3f}'
        })

        # 在指定 episode 保存权重
        if (episode + 1) in [100, 200, 300, 400, 500]:
            save_path = os.path.join(save_dir, f'dqn_model_ep{episode+1}.pth')
            torch.save(agent.q_net.state_dict(), save_path)
            pbar.write(f"✓ 权重已保存: {save_path}")
        
        # 提前通关条件 (CartPole 满分是 500 分)
        if episode_reward >= 500:
            pbar.write(f"🚀 Episode {episode+1}: 满分通关！Reward=500")

    env.close()

if __name__ == "__main__":
    main()