import argparse
import logging
import os
import time

import numpy as np
import torch
import wandb

from torchrl.misc import (get_replay_buffer, prepare_output_dir,
                          initialize_logger, render_env)

torch.backends.cudnn.benchmark = True


class RLTrainer:
    def __init__(self,
                 policy,
                 env,
                 device,
                 args,
                 test_env=None,
                 wandb_turn_on=False,
                 wandb_configs=None):
        # yapf: disable
        if isinstance(args, dict):
            _args = args
            args = policy.__class__.get_argument(RLTrainer.get_argument())
            args = args.parse_args([])
            for k, v in _args.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                else:
                    raise ValueError(f"{k} is invalid parameter.")

        self._set_from_args(args)
        self._policy = policy
        self._env = env
        self._test_env = self._env if test_env is None else test_env
        self.device = device

        # prepare log directory
        self._output_dir = prepare_output_dir(
            args=args, user_specified_dir=self._logdir,
            suffix=f'{self._policy.policy_name}_{args.dir_suffix}')
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level),
            output_dir=self._output_dir)

        self._log_wandb = wandb_turn_on
        self._monitor_gym = False
        self._wandb_dict = None
        if wandb_turn_on:
            self._wandb_dict = {}
            self._monitor_update_interval = 1e4
            wandb.init(entity=wandb_configs['entity'],
                       project=wandb_configs['project'],
                       name=wandb_configs['run_name'])
            self.wandb_configs = wandb_configs
            self._monitor_gym = self.wandb_configs['monitor_gym']

    @staticmethod
    def get_argument(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env.')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process')
        parser.add_argument('--save-model-interval', type=int, default=int(1e4),
                            help='Interval to save model')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results')
        parser.add_argument('--normalize-obs', action='store_true',
                            help='Normalize observation')
        parser.add_argument('--logdir', type=str, default='results',
                            help='Output directory')
        # test settings
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate trained model')
        parser.add_argument('--test-interval', type=int, default=int(1e4),
                            help='Interval to evaluate trained model')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process')
        parser.add_argument('--test-episodes', type=int, default=5,
                            help='Number of episodes to evaluate at once')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation')
        parser.add_argument('--show-test-images', action='store_true',
                            help='Show input images to neural networks when an episode finishes')
        parser.add_argument('--save-test-movie', action='store_true',
                            help='Save rendering results')
        # replay buffer
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level')
        # yapf: enable
        return parser

    def _set_from_args(self, args):
        # experiment settings
        self._max_steps = int(args.max_steps)
        self._episode_max_steps = (args.episode_max_steps
                                   if args.episode_max_steps is not None else
                                   args.max_steps)
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    def __call__(self):
        total_steps = 0
        episode_steps = 0
        episode_return = 0
        episode_start_time = time.perf_counter()
        n_episode = 0

        replay_buffer = get_replay_buffer(self._policy, self._env,
                                          self._use_prioritized_rb,
                                          self._use_nstep_rb, self._n_step)

        obs = self._env.reset()

        while total_steps < self._max_steps:
            if total_steps < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                action = self._policy.get_action(obs)

            next_obs, reward, done, _ = self._env.step(action)
            episode_steps += 1
            episode_return += reward
            total_steps += 1

            done_flag = done
            if (hasattr(self._env, "_max_episode_steps")
                    and episode_steps == self._env._max_episode_steps):
                done_flag = False
            replay_buffer.add(obs=obs,
                              act=action,
                              next_obs=next_obs,
                              rew=reward,
                              done=done_flag)
            obs = next_obs

            if done or episode_steps == self._episode_max_steps:
                replay_buffer.on_episode_end()
                obs = self._env.reset()

                n_episode += 1
                fps = episode_steps / (time.perf_counter() -
                                       episode_start_time)
                self.logger.info(
                    "Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}"
                    .format(n_episode, total_steps, episode_steps,
                            episode_return, fps))
                if self._log_wandb:
                    self._wandb_dict['Common/training_return'] = episode_return

                episode_steps = 0
                episode_return = 0
                episode_start_time = time.perf_counter()

            if total_steps < self._policy.n_warmup:
                continue

            if total_steps % self._policy.update_interval == 0:
                samples = self._to_torch_tensor(
                    replay_buffer.sample(self._policy.batch_size))

                # train policy
                outputs = self._policy.train(samples['obs'],
                                             samples['act'],
                                             samples['next_obs'],
                                             samples['rew'],
                                             samples['done'],
                                             wandb_dict=self._wandb_dict)
                if self._log_wandb and total_steps % self._save_summary_interval == 0:
                    wandb.log(self._wandb_dict)

                if self._use_prioritized_rb:
                    td_error = self._policy.compute_td_error(
                        samples["obs"], samples["act"], samples["next_obs"],
                        samples["rew"],
                        np.array(samples["done"], dtype=np.float32))
                    replay_buffer.update_priorities(samples["indexes"],
                                                    np.abs(td_error) + 1e-6)

                if self._monitor_gym and total_steps % self._monitor_update_interval == 0:
                    self._log_gym_to_wandb(self.wandb_configs['gif_header'] +
                                           str(total_steps) + '.gif')

            if total_steps % self._test_interval == 0:
                avg_test_return = self.evaluate_policy(total_steps)
                self.logger.info(
                    "Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes"
                    .format(total_steps, avg_test_return, self._test_episodes))
                if self._log_wandb:
                    self._wandb_dict[
                        'Common/average_test_return'] = avg_test_return
                    self._wandb_dict['Common/fps'] = fps

            if self._log_wandb:
                wandb.log(self._wandb_dict)

    def _log_gym_to_wandb(self, filename):
        # obtain gym.env from rllab.env
        render_env(self._env,
                   path=self.wandb_configs['gif_dir'],
                   filename=filename)
        if self._log_wandb:
            full_fn = os.path.join(os.getcwd(), self.wandb_configs['gif_dir'],
                                   filename)
            wandb.log({"video": wandb.Video(full_fn, fps=60, format="gif")})

    def evaluate_policy(self, step):
        if self._normalize_obs:
            self._test_env.normalizer.set_params(
                *self._env.normalizer.get_params())
        avg_test_return = 0.
        if self._save_test_path:
            replay_buffer = get_replay_buffer(self._policy,
                                              self._test_env,
                                              size=self._episode_max_steps)

        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            obs = self._test_env.reset()
            for _ in range(self._episode_max_steps):
                action = self._policy.get_action(obs, test=True)
                next_obs, reward, done, _ = self._test_env.step(action)
                if self._save_test_path:
                    replay_buffer.add(obs=obs,
                                      act=action,
                                      next_obs=next_obs,
                                      rew=reward,
                                      done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))
                elif self._show_test_progress:
                    self._test_env.render()
                episode_return += reward
                obs = next_obs
                if done:
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                step, i, episode_return)
            if self._save_test_path:
                save_path(
                    replay_buffer._encode_sample(
                        np.arange(self._episode_max_steps)),
                    os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return

        return avg_test_return / self._test_episodes

    def _to_torch_tensor(self, samples):
        torch_samples = {}
        torch_samples['obs'] = torch.from_numpy(samples['obs']).to(self.device)
        torch_samples['act'] = torch.from_numpy(samples['act']).to(self.device)
        torch_samples['next_obs'] = torch.from_numpy(samples['next_obs']).to(
            self.device)
        torch_samples['rew'] = torch.from_numpy(samples['rew']).to(self.device)
        torch_samples['done'] = torch.tensor(samples['done'],
                                             dtype=torch.float32).to(
                                                 self.device)

        return torch_samples
