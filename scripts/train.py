"""
统一训练入口

使用方式:
    python scripts/train.py --algo dqn --env cartpole
    python scripts/train.py --algo ppo --env lunarlander
"""

import argparse
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# 支持的算法和环境组合
ALGO_ENV_MAP = {
    "dqn": ["cartpole", "lunarlander"],
    "policy_gradient": ["cartpole"],
    "actor_critic": ["cartpole"],
    "ppo": ["cartpole", "lunarlander"],
}


def train_dqn(env_name):
    """运行 DQN 训练"""
    if env_name == "cartpole":
        from algorithms.dqn.cartpole.dqn_cartpole import main
        main()
    elif env_name == "lunarlander":
        from algorithms.dqn.lunarlander.dqn_lunarlander import main
        main()
    else:
        print(f"Error: DQN on {env_name} not implemented yet.")


def train_policy_gradient(env_name):
    """运行 Policy Gradient 训练"""
    if env_name == "cartpole":
        from algorithms.policy_gradient.cartpole.policy_gradient_cartpole import main
        main()
    else:
        print(f"Error: Policy Gradient on {env_name} not implemented yet.")


def train_actor_critic(env_name):
    """运行 Actor-Critic 训练"""
    if env_name == "cartpole":
        from algorithms.actor_critic.cartpole.actor_critic_cartpole import main
        main()
    else:
        print(f"Error: Actor-Critic on {env_name} not implemented yet.")


def train_ppo(env_name):
    """运行 PPO 训练"""
    if env_name == "cartpole":
        from algorithms.ppo.cartpole.ppo_cartpole import main
        main()
    elif env_name == "lunarlander":
        from algorithms.ppo.lunarlander.ppo_lunarlander import main
        main()
    else:
        print(f"Error: PPO on {env_name} not implemented yet.")


# 算法分发器
ALGO_FUNC = {
    "dqn": train_dqn,
    "policy_gradient": train_policy_gradient,
    "actor_critic": train_actor_critic,
    "ppo": train_ppo,
}


def main():
    parser = argparse.ArgumentParser(description="RL 统一训练入口")
    parser.add_argument("--algo", type=str, required=True,
                        choices=list(ALGO_FUNC.keys()),
                        help="算法名称")
    parser.add_argument("--env", type=str, required=True,
                        help="环境名称 (cartpole, lunarlander)")

    args = parser.parse_args()

    # 验证算法和环境组合
    if args.env not in ALGO_ENV_MAP.get(args.algo, []):
        print(f"Error: {args.algo} on {args.env} is not supported.")
        print(f"Supported environments for {args.algo}: {ALGO_ENV_MAP.get(args.algo, [])}")
        sys.exit(1)

    # 执行训练
    print(f"\n{'='*50}")
    print(f" 开始训练: {args.algo.upper()} on {args.env.capitalize()}")
    print(f"{'='*50}\n")

    ALGO_FUNC[args.algo](args.env)


if __name__ == "__main__":
    main()
