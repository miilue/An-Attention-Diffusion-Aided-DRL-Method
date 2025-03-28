import os
import sys
import pprint
import argparse
from os import path
from datetime import datetime

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DiscreteSACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

# 将当前文件所在目录的父目录添加到Python解释器的搜索路径，以便在运行时能够导入该父目录中的模块
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from env import make_env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="AaaS")
    parser.add_argument('--algorithm', type=str, default='SAC')
    parser.add_argument('--algorithm-fix', type=str, default='default')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)  # 策略评估的轮数
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--step-per-epoch", type=int, default=100)  # 每个epoch收集的transitions数目
    parser.add_argument("--step-per-collect", type=int, default=1000)  # 在网络更新之前，收集器收集的transitions数目
    parser.add_argument("--update-per-step", type=float, default=1)  # 每次collect，策略更新update-per-step * step-per-collect次
    parser.add_argument("--reward-threshold", type=float, default=None)  # 奖励阈值
    parser.add_argument("--gamma", type=float, default=0.95)  # 奖励折扣率[0, 1]，默认0.99
    parser.add_argument("--n-step", type=int, default=3)  # n-step TD，默认1
    parser.add_argument('--rew-norm', type=int, default=0)  # 标准化reward，默认False
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument('--wd', type=float, default=1e-4)  # weight_decay

    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.01)  # 渲染连续帧之间的休眠时间

    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action="store_true", default=False)  # 是否监视（不训练，只测试）

    # for sac
    parser.add_argument("--tau", type=float, default=0.005)  # target network 的 soft update，默认0.005
    parser.add_argument("--alpha", type=float, default=0.05)  # 熵正则化系数，默认0.2
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--auto-alpha", action="store_true", default=False)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)  # alpha的学习率（auto tune）

    return parser.parse_known_args()[0]


def test_sac(args=get_args()):
    env, train_envs, test_envs = make_env(args.task, args.training_num, args.test_num)
    # env = gym.make(args.task)
    # train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # log
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    root = path.dirname(path.dirname(path.abspath(__file__)))
    log_path = os.path.join(root, args.logdir, args.task, args.algorithm, args.algorithm_fix, time_now)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        if args.reward_threshold:
            return mean_rewards >= args.reward_threshold
        return False

    # model
    # create actor
    net = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Mish,
        device=args.device
    )
    actor = Actor(
        net,
        args.action_shape,
        softmax_output=False,
        device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=args.wd)

    # create critic
    net_c1 = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Mish,
        device=args.device
    )
    critic1 = Critic(
        net_c1,
        last_size=args.action_shape,
        device=args.device
    ).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr, weight_decay=args.wd)
    net_c2 = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Mish,
        device=args.device
    )
    critic2 = Critic(
        net_c2,
        last_size=args.action_shape,
        device=args.device
    ).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr, weight_decay=args.wd)

    # better not to use auto alpha in CartPole
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # policy
    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        reward_normalization=args.rew_norm
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
    )
    test_collector = Collector(policy, test_envs)

    # trainer
    if not args.watch:
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=args.update_per_step,
            test_in_train=False,
        ).run()
        pprint.pprint(result)
        print('-------------------------------')

    # watch its performance
    if __name__ == "__main__":
        np.random.seed(args.seed)
        env, _, _ = make_env(args.task, if_test=True)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        pprint.pprint(result)


if __name__ == "__main__":
    test_sac(get_args())

