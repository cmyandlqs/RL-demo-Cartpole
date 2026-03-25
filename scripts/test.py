"""
统一测试入口

使用方式:
    python scripts/test.py --algo dqn --env cartpole --model outputs\dqn\cartpole\checkpoints\dqn_model_ep902_reward479.pth
    python scripts/test.py --algo ppo --env cartpole --model outputs/ppo/cartpole/checkpoints/model.pth

    --seed: 设置随机种子（可选，用于确定性测试）
    --episodes: 测试回合数，默认 5
    --no-render: 不显示可视化窗口（headless 模式）
"""

import argparse
import os
import sys
import gymnasium as gym
import torch
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_dqn_cartpole(model_path, num_episodes=5, seed=None, render=True):
    """测试 DQN on CartPole"""
    from algorithms.dqn.cartpole.dqn_cartpole import QNetwork

    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim)
    q_net.load_state_dict(torch.load(model_path))
    q_net.eval()
    print(f" 已加载模型: {model_path}")

    rewards = []
    for episode in range(num_episodes):
        state, info = env.reset(seed=seed + episode if seed is not None else None)
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = q_net(state_tensor)
            action = q_values.argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)
        print(f" Episode {episode+1}: Reward = {episode_reward}")

    env.close()
    print(f"\n 测试完成! 平均奖励: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")


def test_dqn_lunarlander(model_path, num_episodes=5, seed=None, render=True):
    """测试 DQN on LunarLander"""
    print("Error: DQN on LunarLander not implemented yet.")
    sys.exit(1)


def test_ppo_cartpole(model_path, num_episodes=5, seed=None, render=True):
    """测试 PPO on CartPole"""
    print("Error: PPO on CartPole not implemented yet.")
    sys.exit(1)


def test_ppo_lunarlander(model_path, num_episodes=5, seed=None, render=True):
    """测试 PPO on LunarLander"""
    print("Error: PPO on LunarLander not implemented yet.")
    sys.exit(1)


# 测试函数映射
TEST_FUNC = {
    ("dqn", "cartpole"): test_dqn_cartpole,
    ("dqn", "lunarlander"): test_dqn_lunarlander,
    ("ppo", "cartpole"): test_ppo_cartpole,
    ("ppo", "lunarlander"): test_ppo_lunarlander,
}


def main():
    parser = argparse.ArgumentParser(description="RL 统一测试入口")
    parser.add_argument("--algo", type=str, required=True,
                        choices=["dqn", "policy_gradient", "actor_critic", "ppo"],
                        help="算法名称")
    parser.add_argument("--env", type=str, required=True,
                        help="环境名称 (cartpole, lunarlander)")
    parser.add_argument("--model", type=str, required=True,
                        help="模型文件路径")
    parser.add_argument("--episodes", type=int, default=5,
                        help="测试回合数 (默认: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（用于确定性测试）")
    parser.add_argument("--no-render", action="store_true",
                        help="不显示可视化窗口")

    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f" Error: 模型文件不存在: {args.model}")
        sys.exit(1)

    # 执行测试
    print(f"\n{'='*50}")
    print(f" 开始测试: {args.algo.upper()} on {args.env.capitalize()}")
    print(f"{'='*50}\n")

    key = (args.algo, args.env)
    if key not in TEST_FUNC:
        print(f" Error: {args.algo} on {args.env} 测试函数未实现")
        sys.exit(1)

    TEST_FUNC[key](
        model_path=args.model,
        num_episodes=args.episodes,
        seed=args.seed,
        render=not args.no_render
    )


if __name__ == "__main__":
    main()
