from torchrl.misc import get_replay_buffer


class Trainer:
    def __init__(self):
        pass

    def __call__(self):
        replay_buffer = get_replay_buffer(self._policy, self._env,
                                          self._use_prioritized_rb,
                                          self._use_nstep_rb, self._nstep)

        obs = self._env.reset()

        for step in self._max_steps:
            if step < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                action = self._policy.get_action(obs)

            next_obs, reward, done, _ = self._env.step(action)

            replay_buffer.add(obs=obs,
                              act=action,
                              next_obs=next_obs,
                              rew=reward,
                              done=done_flag)

            obs = next_obs

            if step % self._policy.update_interval == 0:
                samples = replay_buffer.sample(self._policy.batch_size)
                self._policy.train(samples['obs'], samples['act'],
                                   samples['next_obs'], samples['rew'],
                                   np.array(samples['done'], dtype=np.float32))
