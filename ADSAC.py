import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import pprint
import argparse
from os import path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from env import make_env
from model import *
from model.diffusion import Diffusion
from model.model import DoubleCritic
from policy import DiffusionSAC


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="AaaS")
    parser.add_argument('--algorithm', type=str, default='SAC')
    parser.add_argument('--algorithm-fix', type=str, default='Diffusion')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--training-num", type=int, default=1)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--step-per-epoch", type=int, default=100)
    parser.add_argument("--step-per-collect", type=int, default=1000)
    parser.add_argument("--update-per-step", type=float, default=1)
    parser.add_argument("--reward-threshold", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=256)
    parser.add_argument('--wd', type=float, default=1e-4)

    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.1)

    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        )
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action="store_true", default=False)

    # for sac
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--auto-alpha", action="store_true", default=False)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)

    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--n-timesteps', type=int, default=5)
    parser.add_argument('--beta-schedule', type=str, default='vp', choices=['linear', 'cosine', 'vp'])
    parser.add_argument('--pg-coef', type=float, default=1.)

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.6)
    parser.add_argument('--prior-beta', type=float, default=0.4)

    parser.add_argument('--attention', action='store_true', default=False)

    return parser.parse_known_args()[0]


def test_sac_diffusion(args=get_args()):
    print('device: ', args.device)
    env, train_envs, test_envs = make_env(args.task, args.training_num, args.test_num)
    # env = gym.make(args.task)
    # train_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    args.max_action = 1.

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
    if args.attention == True:
        net = Diffusion_Attention(
            state_dim=args.state_shape,
            action_dim=args.action_shape,
            activation=nn.Mish,
            device=args.device
            )
    else:
        net = Diffusion_Net(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        activation=nn.Mish,
        device=args.device
        )

    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=net,
        max_action=args.max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        device=args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=args.wd)

    # create critic
    critic = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        hidden_dim=args.hidden_sizes,
        activation=nn.Mish
    ).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=args.wd)

    # better not to use auto alpha in CartPole
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # policy
    policy = DiffusionSAC(
        actor,
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        torch.distributions.Categorical,
        args.device,
        alpha=args.alpha,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        pg_coef=args.pg_coef,
        action_space=env.action_space
    )

    # load a previous policy
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # buffer
    if args.prioritized_replay:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,
            beta=args.prior_beta,
        )
    else:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )

    # collector
    train_collector = Collector(policy, train_envs, buffer)
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
            test_in_train=False
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
    test_sac_diffusion(get_args())
