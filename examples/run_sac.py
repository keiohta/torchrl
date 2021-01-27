import gym
import torch

from torchrl.algos import SAC
from torchrl.experiments import RLTrainer as Trainer
from torchrl.experiments import WandB

if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = SAC.get_argument(parser)
    parser = WandB.get_argument(parser)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-unit", type=int, default=256)
    parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    parser.add_argument('--no_cuda', action='store_true')
    parser.set_defaults(batch_size=256)
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(max_steps=3e6)
    parser.set_defaults(save_model_interval=10000)
    parser.set_defaults(max_steps=100000)
    parser.set_defaults(memory_capacity=10000)
    parser.set_defaults(dump_data=True)
    args = parser.parse_args()

    exp_name = f"L{args.n_layer}-U{args.n_unit}"
    args.dir_suffix = exp_name

    # setup wandb
    wandb_configs = None
    if args.wandb_turn_on:
        args.wandb_run_name = f"sac_pendulum_{exp_name}"
        wandb_configs = {}
        for arg in vars(args):
            if 'wandb' in arg:
                wandb_configs[arg.replace('wandb_', '')] = vars(args)[arg]

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    units = [args.n_unit for _ in range(args.n_layer)]

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = SAC(state_shape=env.observation_space.shape,
                 action_dim=env.action_space.high.size,
                 device=device,
                 actor_units=units,
                 critic_units=units,
                 memory_capacity=args.memory_capacity,
                 max_action=env.action_space.high[0],
                 batch_size=args.batch_size,
                 n_warmup=args.n_warmup,
                 alpha=args.alpha,
                 auto_alpha=args.auto_alpha)
    trainer = Trainer(policy,
                      env,
                      device,
                      args,
                      test_env=test_env,
                      wandb_turn_on=args.wandb_turn_on,
                      wandb_configs=wandb_configs)
    trainer()
