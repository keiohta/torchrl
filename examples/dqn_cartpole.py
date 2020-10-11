import gym

from torchrl.algos import DQN
from torchrl.experiments import RLTrainer

if __name__ == '__main__':
    # yapf: disable
    parser = RLTrainer.get_argument()
    parser = DQN.get_argument(parser)
    parser.set_defaults(test_interval=2000)
    parser.set_defaults(max_steps=100000)
    parser.set_defaults(gpu=-1)
    parser.set_defaults(n_warmup=500)
    parser.set_defaults(batch_size=32)
    parser.set_defaults(memory_capacity=int(1e4))
    parser.add_argument('--env-name', type=str, default="CartPole-v0")
    parser.add_argument('--wandb_turn_on', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default='sff1019')
    parser.add_argument('--wandb_project', type=str,
                        default='reinforcement_learning_algorithms')
    parser.add_argument('--wandb_run_name', type=str, default='torchrl_dqn_cartpole')
    parser.add_argument('--wandb_monitor_gym', action='store_true')
    parser.add_argument('--wandb_gif_dir', type=str, default='logs/dqn_cartpole')
    parser.add_argument('--wandb_gif_header', type=str, default='')
    args = parser.parse_args()
    # yapf: enable

    # setup wandb
    if args.wandb_turn_on:
        wandb_configs = {}
        for arg in vars(args):
            if 'wandb' in arg:
                wandb_configs[arg.replace('wandb_', '')] = vars(args)[arg]

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = DQN(state_shape=env.observation_space.shape,
                 action_dim=env.action_space.n,
                 target_update_interval=300,
                 discount=0.99,
                 memory_capacity=args.memory_capacity,
                 batch_size=args.batch_size,
                 n_warmup=args.n_warmup)
    trainer = RLTrainer(policy,
                        env,
                        args,
                        test_env=test_env,
                        wandb_turn_on=args.wandb_turn_on,
                        wandb_configs=wandb_configs)
    trainer()
