# torchrl

torchrl is a deep reinforcement and inverse reinforcement learning tool using PyTorch.
This project is a reimplementation of [tf2rl](https://github.com/keiohta/tf2rl), a deep reinforcement learning algorithms using Tensorflow 2.x, using pytorch.
Hence, all the kudos go to tf2rl.

# Algorithms

|Algorithm|Category|Check Results|
|:----:|:---:|:----:|
|DQN|Model-free Off-policy RL| Mid-way|
|SAC|Model-free Off-policy RL| x|
|GAIL|Imitation Learning| x|

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
