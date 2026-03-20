import torch
import gymnasium as gym

print("==============================")
print(f"PyTorch 版本: {torch.__version__}")
print(f"是否支持 GPU: {torch.cuda.is_available()}")
print("==============================")

# 1. 创建 CartPole 环境，render_mode="human" 会弹出一个窗口让我们看到画面
env = gym.make("CartPole-v1", render_mode="human")

# 2. 初始化环境，拿到小车和杆子的初始状态
state, info = env.reset()
print(f"初始状态 (State) 维度: {state.shape}")
print(f"初始状态数值: {state}")

# 3. 让一个小傻瓜 Agent 随机乱动 100 步
for _ in range(500):
    # env.action_space.sample() 表示在这个游戏里纯随机选一个动作（0或1）
    random_action = env.action_space.sample() 
    
    # 将动作喂给环境，获取下一步的反馈
    next_state, reward, terminated, truncated, info = env.step(random_action)
    
    # 只要满足条件（比如杆子倒了），就立刻重置环境，重新开始下一局
    if terminated or truncated:
        env.reset()

# 4. 结束后关闭环境窗口
env.close()
print("环境交互测试成功！小车运行正常。")