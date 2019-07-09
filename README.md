## Research_RL
This is a repo for research purpose.

I am mainly relying on my other repo specialised in RL: [TF_RL](https://github.com/Rowing0914/TF_RL).
so, please check that as well. 

## Usage
```shell
# Q-learning with GP Performance Prediction model on cartpole
$ python3.6 src.Qlearn_GP_batch.py

# DDPG with GP Performance Prediction model on cartpole
$ python3.6 src.DQN_GP_batch_cartpole.py

# DDPG with GP Performance Prediction model on MuJoCo
$ python3.6 src.DDPG_GP.py
```

## Directory
```shell
.
├── logs: Training log
├── result: Resulting graphs
├── src: Training scripts
└── utils: Supporting scripts used in training scripts

```
