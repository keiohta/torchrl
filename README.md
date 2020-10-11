# torchrl

torchrl is a deep reinforcement and inverse reinforcement learning tool using PyTorch.

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

This project is heavily referencing [tf2rl](https://github.com/keiohta/tf2rl), a deep reinforcement learning algorithms using Tensorflow 2.x.

# Quick commands (removing soon)
```
# to render on server
$ xvfb-run -s "-screen 0 1400x900x24" python [file_name].py
```
