import argparse


class WandB:
    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # yapf: disable
        parser.add_argument('--env-name', type=str, default="Pendulum-v0")
        parser.add_argument('--wandb_turn_on', action='store_true')
        parser.add_argument('--wandb_entity', type=str, default='ohtake_i')
        parser.add_argument('--wandb_project', type=str, default='torchrl')
        parser.add_argument('--wandb_run_name', type=str, default='torchrl_sac_pendulum')
        parser.add_argument('--wandb_monitor_gym', action='store_true')
        parser.add_argument('--wandb_gif_dir', type=str, default='logs/dqn_cartpole')
        parser.add_argument('--wandb_gif_header', type=str, default='')
        # yapf: enable

        return parser
