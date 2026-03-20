import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================
# 复用 QNetwork 结构 (必须与训练时完全一致)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==========================================
# 加载模型并渲染
# ==========================================
def load_and_render(model_path, num_episodes=5):
    # 创建环境，render_mode="human" 会弹出可视化窗口
    env = gym.make("CartPole-v1", render_mode="human")

    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n             # 2

    # 初始化网络
    q_net = QNetwork(state_dim, action_dim)

    # 加载训练好的权重
    q_net.load_state_dict(torch.load(model_path))
    q_net.eval()  # 切换到评估模式（关闭 dropout 等）
    print(f"✓ 已加载模型: {model_path}")

    # 测试指定局数
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # 将状态转为 tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # 用模型预测，不需要梯度
            with torch.no_grad():
                q_values = q_net(state_tensor)

            # 选择 Q 值最大的动作（纯贪婪，无随机探索）
            action = q_values.argmax().item()

            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

        print(f"Episode {episode+1}: Reward = {episode_reward}")

    env.close()
    print("\n测试完成！")


if __name__ == "__main__":
    # 指定要加载的模型路径
    model_path = "checkpoints/dqn_model_ep300.pth"

    # 如果还没有训练，可以用这个路径测试（会报错，提醒你先训练）
    import os
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 python dqn_cartpole.py 训练模型")
        exit(1)

    load_and_render(model_path, num_episodes=5)
