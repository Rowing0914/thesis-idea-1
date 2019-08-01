"""
Note:
    This is a test script to see if we can safely invoke the `create_model` function
    which creates GP model for the performance prediction!
"""

import argparse
import tensorflow_probability as tfp
from tf_rl.common.utils import *
from tf_rl.common.networks import CartPole as Model
from tf_rl.agents.DQN import DQN_cartpole
from src.common import RBFKernelFn
from src.common.gp_models import create_model

tfd = tfp.distributions

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="CartPole-v0", help="game env Atari games or CartPole")
parser.add_argument("--seed", default=123, help="seed of randomness")
parser.add_argument("--loss_fn", default="huber", help="types of loss function: MSE or huber")
parser.add_argument("--grad_clip_flg", default="", help="gradient clippings: by_value or norm or nothing")
parser.add_argument("--num_frames", default=30000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=1, type=int, help="a frequency of training occurring in training phase")
parser.add_argument("--eval_interval", default=2500, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=5000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=100, type=int, help="when to start learning")
parser.add_argument("--sync_freq", default=1000, type=int, help="frequency of updating a target model")
parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--tau", default=1e-2, type=float, help="soft update tau")
parser.add_argument("--ep_start", default=1.0, type=float, help="initial value of epsilon")
parser.add_argument("--ep_end", default=0.02, type=float, help="final value of epsilon")
parser.add_argument("--lr_start", default=0.0025, type=float, help="initial value of lr")
parser.add_argument("--lr_end", default=0.00025, type=float, help="final value of lr")
parser.add_argument("--decay_steps", default=3000, type=int, help="a period for annealing a value(epsilon or beta)")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.goal = 195
params.test_episodes = 20

now = datetime.datetime.now()

if params.debug_flg:
    params.log_dir = "../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-test/"
    params.model_dir = "../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-test/"
else:
    params.log_dir = "../logs/logs/{}".format(params.env_name)
    params.model_dir = "../logs/models/{}".format(params.env_name)

# init global time-step
global_timestep = tf.train.get_or_create_global_step()

# instantiate annealing funcs for ep and lr
anneal_lr = tf.compat.v1.train.polynomial_decay(params.lr_start, global_timestep, params.decay_steps, params.lr_end)

optimizer = tf.compat.v1.train.RMSPropOptimizer(anneal_lr, 0.99, 0.0, 1e-6)
loss_fn = create_loss_func(params.loss_fn)
grad_clip_fn = gradient_clip_fn(flag=params.grad_clip_flg)

env = MyWrapper(gym.make(params.env_name))
agent = DQN_cartpole(Model, optimizer, loss_fn, grad_clip_fn, env.action_space.n, params)
init_state = env.reset() # reset
agent.predict(init_state) # burn the format of the input matrix to get the weight matrices!!
gp_model = create_model(weights=agent.main_model.get_weights(), kernel_fn=RBFKernelFn)