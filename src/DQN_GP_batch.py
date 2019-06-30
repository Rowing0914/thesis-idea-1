import argparse, time
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import *
from tf_rl.common.policy import EpsilonGreedyPolicy_eager
from tf_rl.common.networks import CartPole as Model
from tf_rl.agents.DQN import DQN_cartpole

tfd = tfp.distributions

eager_setup()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="CartPole", help="game env type: Atari or CartPole")
parser.add_argument("--seed", default=123, help="seed of randomness")
parser.add_argument("--loss_fn", default="huber", help="types of loss function: MSE or huber")
parser.add_argument("--grad_clip_flg", default="", help="gradient clippings: by value(by_value) or global norm(norm) or nothing")
parser.add_argument("--num_frames", default=30000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=1, type=int, help="a frequency of training occurring in training phase")
parser.add_argument("--eval_interval", default=2500, type=int, help="a frequency of evaluation occurring in training phase") # temp
parser.add_argument("--memory_size", default=5000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=100, type=int, help="frame number which specifies when to start updating the agent")
parser.add_argument("--sync_freq", default=1000, type=int, help="frequency of updating a target model")
parser.add_argument("--batch_size", default=32, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=10, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor: gamma > 1.0 or negative => does not converge!!")
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

# init global time-step
global_timestep = tf.train.get_or_create_global_step()

# instantiate annealing funcs for ep and lr
anneal_ep = tf.train.polynomial_decay(params.ep_start, global_timestep, params.decay_steps, params.ep_end)
anneal_lr = tf.train.polynomial_decay(params.lr_start, global_timestep, params.decay_steps, params.lr_end)

# prep for training
policy = EpsilonGreedyPolicy_eager(Epsilon_fn=anneal_ep)
optimizer = tf.train.RMSPropOptimizer(anneal_lr, 0.99, 0.0, 1e-6)
replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
loss_fn = create_loss_func(params.loss_fn)
grad_clip_fn = gradient_clip_fn(flag=params.grad_clip_flg)

# create a directory for log/model
params = create_log_model_directory(params, get_alg_name())
now = datetime.datetime.now()
params.log_dir = "../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DQN_GP/"
params.model_dir = "../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DQN_GP/"

summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)

# instantiate agent and env
env = MyWrapper(gym.make("CartPole-v0"))
agent = DQN_cartpole(Model, optimizer, loss_fn, grad_clip_fn, env.action_space.n, params)

# set seed
env.seed(params.seed)
tf.random.set_random_seed(params.seed)



class RBFKernelFn(tf.keras.layers.Layer):
	# https://medium.com/tensorflow/regression-with-probabilistic-layers-in-tensorflow-probability-e46ff5d37baf
	def __init__(self, **kwargs):
		super(RBFKernelFn, self).__init__(**kwargs)
		dtype = kwargs.get('dtype', None)

		self._amplitude = self.add_variable(
			initializer=tf.constant_initializer(0),
			dtype=dtype,
			name='amplitude')

		self._length_scale = self.add_variable(
			initializer=tf.constant_initializer(0),
			dtype=dtype,
			name='length_scale')

	def call(self, x):
		# Never called -- this is just a layer so it can hold variables
		# in a way Keras understands.
		return x

	@property
	def kernel(self):
		return tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
			amplitude=tf.nn.softplus(0.1 * self._amplitude),
			length_scale=tf.nn.softplus(5. * self._length_scale)
		)

def create_model():
	dtype = np.float64
	num_inducing_points = 40
	loss = lambda y, rv_y: rv_y.variational_loss(y)

	model = tf.keras.Sequential([
		tf.keras.layers.InputLayer(input_shape=[658], dtype=dtype),
		tf.keras.layers.Dense(379, kernel_initializer='ones', use_bias=False),
		tf.keras.layers.Dense(169, kernel_initializer='ones', use_bias=False),
		tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
		tfp.layers.VariationalGaussianProcess(
			num_inducing_points=num_inducing_points,
			kernel_provider=RBFKernelFn(dtype=dtype),
			event_shape=[1],
			# inducing_index_points_initializer=tf.constant_initializer(
			# 	np.linspace(*x_range, num=num_inducing_points,
			# 				dtype=x.dtype)[..., np.newaxis]),
			unconstrained_observation_noise_variance_initializer=(
				tf.constant_initializer(np.array(0.54).astype(dtype))),
		),
	])
	model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss=loss)
	return model


gp_model = create_model()
batch_size = 32
num_epochs = 200
num_sample = 100  # number of sampling


get_ready(agent.params)
time_buffer = list()
global_timestep = tf.train.get_or_create_global_step()
log = logger(agent.params)

with summary_writer.as_default():
	with tf.contrib.summary.always_record_summaries():
		policies, scores = list(), list()
		_max, _min, _means = list(), list(), list()
		for i in itertools.count():
			state = env.reset()
			total_reward = 0
			start = time.time()
			cnt_action = list()
			done = False
			while not done:
				action = policy.select_action(agent, state)
				next_state, reward, done, info = env.step(action)
				replay_buffer.add(state, action, reward, next_state, done)

				global_timestep.assign_add(1)
				total_reward += reward
				state = next_state
				cnt_action.append(action)

				# for evaluation purpose
				if global_timestep.numpy() % agent.params.eval_interval == 0:
					agent.eval_flg = True

				if (global_timestep.numpy() > agent.params.learning_start) and (
						global_timestep.numpy() % agent.params.train_interval == 0):
					states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)

					loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

				# synchronise the target and main models by hard
				if (global_timestep.numpy() > agent.params.learning_start) and (
						global_timestep.numpy() % agent.params.sync_freq == 0):
					agent.manager.save()
					agent.target_model.set_weights(agent.main_model.get_weights())

			"""
			===== After 1 Episode is Done =====
			"""
			tf.contrib.summary.scalar("Return", total_reward, step=i)
			tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
			if i >= agent.params.reward_buffer_ep:
				tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)
			tf.contrib.summary.histogram("taken actions", cnt_action, step=i)

			# store the episode reward
			reward_buffer.append(total_reward)
			time_buffer.append(time.time() - start)

			if agent.main_model.get_weights() != []: # initially the weights are empty, so exclude it explicitly
				scores.append(total_reward) # save the reward which has the same amount as `policies` buffer
				weights_vec = np.array([]) # save the weights used in the Q-net
				for weight in agent.main_model.get_weights():
					weights_vec = np.concatenate([weights_vec, weight.flatten()])
				policies.append(weights_vec.tolist())

			# === Train the GP Net ===
			if len(policies) > batch_size:
				history = gp_model.fit(np.array(policies), np.array(scores), batch_size=batch_size, epochs=num_epochs,
									   verbose=False)
				sample_ = gp_model(weights_vec[np.newaxis, ...]).sample(num_sample).numpy() # use the latest weights_vec to see the accuracy
				print("Ep: {} | Return: {} | Mean Est Return: {:.2f} | Mean Loss: {:.3f}".format(
					i, total_reward, sample_.mean(), np.mean(history.history["loss"]))
				)
				tf.contrib.summary.scalar("Estimated Return", sample_.mean(), step=i)
				_max.append(sample_.max())
				_min.append(sample_.min())
				_means.append(sample_.mean())


			if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
				log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss),
							policy.current_epsilon(), cnt_action)
				time_buffer = list()

			if agent.eval_flg:
				test_Agent(agent, env)
				agent.eval_flg = False

			# check the stopping condition
			if global_timestep.numpy() > agent.params.num_frames:
				print("=== Training is Done ===")
				test_Agent(agent, env, n_trial=agent.params.test_episodes)
				env.close()
				break

		# === Visualisation Part ===
		_min, _max, _means = np.array(_min), np.array(_max), np.array(_means)
		plt.title("Starting at {} Ep".format(batch_size))
		plt.plot(scores, label="Return")
		plt.plot(_means, label="Est Return")
		plt.fill_between(np.arange(len(_min)), _min, _max, facecolor='blue', alpha=0.5)
		plt.legend()
		plt.show()