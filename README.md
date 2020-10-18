![unittest](https://github.com/sff1019/torchrl/workflows/unittest/badge.svg)
# torchrl

torchrl is a deep reinforcement and inverse reinforcement learning tool using PyTorch.
This project is a reimplementation of [tf2rl](https://github.com/keiohta/tf2rl), a deep reinforcement learning algorithms using Tensorflow 2.x, using pytorch.
Hence, all the kudos go to tf2rl.

# Algorithms

## Overview

|Algorithm|Category|Check Results|
|:----:|:---:|:----:|
|BC|Imitation Learning| x|
|DDPG|Model-free Off-policy RL| x |
|DQN|Model-free Off-policy RL| Mid-way|
|GAIL|Imitation Learning| x|
|SAC|Model-free Off-policy RL| x|

## Algorithm Reference

- Model-free Off-policy RL
  - DDPG: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
  - DQN: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
  - SAC: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- Imitation Learning
  - BC: [Model-Free Imitation Learning with Policy Optimization](https://arxiv.org/abs/1605.08478)
  - GAIL: [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)

# Installation

Currently only available for local installation
```
# support only through pip

$ git clone git@github.com:sff1019/torchrl.git
$ cd torchrl
$ pip install -r requirements.txt
$ pip install -e .
```

# Reference
- [tf2rl](https://github.com/keiohta/tf2rl): The base of this project


# Quick commands (removing soon)
```
# to render on server
$ xvfb-run -s "-screen 0 1400x900x24" python [file_name].py
```
