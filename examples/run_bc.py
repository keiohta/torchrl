import gym
import torch

from torchrl.algos import DDPG, BC
from torchrl.experiments import IRLTrainer, WandB
from torchrl.experiments.utils import restore_latest_n_traj

if __name__ == '__main__':
    parser = IRLTrainer.get_argument()
    parser = WandB.get_argument(parser)
    parser = BC.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="Pendulum-v0")
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    if args.expert_path_dir is None:
        print("Plaese generate demonstrations first")
        print(
            "python examples/run_sac.py --env-name=Pendulum-v0 --save-test-path --test-interval=50000"
        )
        exit()

    # setup wandb
    wandb_configs = None
    if args.wandb_turn_on:
        wandb_configs = {}
        for arg in vars(args):
            if 'wandb' in arg:
                wandb_configs[arg.replace('wandb_', '')] = vars(args)[arg]

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    units = [400, 300]

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = DDPG(state_shape=env.observation_space.shape,
                  action_dim=env.action_space.high.size,
                  device=device,
                  max_action=env.action_space.high[0],
                  actor_units=units,
                  critic_units=units,
                  n_warmup=10000,
                  batch_size=100)
    irl = BC(state_shape=env.observation_space.shape,
             action_dim=env.action_space.high.size,
             device=device,
             units=units)
    expert_trajs = restore_latest_n_traj(args.expert_path_dir,
                                         n_path=20,
                                         max_steps=1000)
    trainer = IRLTrainer(policy, env, device, args, irl, expert_trajs["obses"],
                         expert_trajs["next_obses"], expert_trajs["acts"],
                         test_env)
    trainer()
