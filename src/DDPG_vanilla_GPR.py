import time
import argparse
import tensorflow as tf
from collections import deque
from tf_rl.common.memory import ReplayBuffer
from tf_rl.common.utils import *
from tf_rl.agents.DDPG import DDPG
from tf_rl.common.params import DDPG_ENV_LIST
from tf_rl.common.networks import DDPG_Actor as Actor, DDPG_Critic as Critic
from utils.common import flatten_weight, test_Agent_DDPG, plotting_fn
from utils.gp_models import sklearn_GP_model

eager_setup()

"""
this is defined in params.py
DDPG_ENV_LIST = {
	"Ant-v2": 3500,
	"HalfCheetah-v2": 7000,
	"Hopper-v2": 1500,
	"Humanoid-v2": 2000,
	"HumanoidStandup-v2": 0, # maybe we don't need this...
	"InvertedDoublePendulum-v2": 6000,
	"InvertedPendulum-v2": 800,
	"Reacher-v2": -6,
	"Swimmer-v2": 40,
	"Walker2d-v2": 2500
}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="Ant-v2", help="Env title")
parser.add_argument("--seed", default=123, type=int, help="seed for randomness")
# parser.add_argument("--num_frames", default=1_000_000, type=int, help="total frame in a training")
parser.add_argument("--num_frames", default=500_000, type=int, help="total frame in a training")
parser.add_argument("--train_interval", default=100, type=int, help="a frequency of training in training phase")
parser.add_argument("--nb_train_steps", default=50, type=int, help="a number of training after one episode")
parser.add_argument("--eval_interval", default=10_000, type=int, help="a frequency of evaluation in training phase")
parser.add_argument("--memory_size", default=100_000, type=int, help="memory size in a training")
parser.add_argument("--learning_start", default=10_000, type=int, help="length before training")
parser.add_argument("--batch_size", default=100, type=int, help="batch size of each iteration of update")
parser.add_argument("--reward_buffer_ep", default=5, type=int, help="reward_buffer size")
parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
parser.add_argument("--soft_update_tau", default=1e-2, type=float, help="soft-update tau ")
parser.add_argument("--L2_reg", default=0.5, type=float, help="magnitude of L2 regularisation")
parser.add_argument("--action_range", default=[-1., 1.], type=list, help="magnitude of L2 regularisation")
parser.add_argument("--debug_flg", default=False, type=bool, help="debug mode or not")
parser.add_argument("--google_colab", default=False, type=bool, help="if you are executing this on GoogleColab")
params = parser.parse_args()
params.test_episodes = 10
params.goal = DDPG_ENV_LIST[params.env_name]

now = datetime.datetime.now()

if params.debug_flg:
    params.log_dir = "../logs/logs/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG/"
    params.model_dir = "../logs/models/" + now.strftime("%Y%m%d-%H%M%S") + "-DDPG/"
else:
    params.log_dir = "../logs/logs/{}".format(params.env_name)
    params.model_dir = "../logs/models/{}".format(params.env_name)

env = gym.make(params.env_name)

# set seed
env.seed(params.seed)
tf.compat.v1.random.set_random_seed(params.seed)

agent = DDPG(Actor, Critic, env.action_space.shape[0], params)
replay_buffer = ReplayBuffer(params.memory_size)
reward_buffer = deque(maxlen=params.reward_buffer_ep)
summary_writer = tf.contrib.summary.create_file_writer(params.log_dir)

init_state = env.reset()  # reset
agent.predict(init_state)  # burn the format of the input matrix to get the weight matrices!!
gp_model = sklearn_GP_model()
num_sample = 100  # number of sampling

get_ready(agent.params)

global_timestep = tf.compat.v1.train.get_or_create_global_step()
time_buffer = deque(maxlen=agent.params.reward_buffer_ep)
log = logger(agent.params)

with summary_writer.as_default():
    # for summary purpose, we put all codes in this context
    with tf.contrib.summary.always_record_summaries():
        policies, scores = list(), list()
        preds, evals = list(), list()
        for i in itertools.count():
            state = env.reset()
            total_reward = 0
            start = time.time()
            agent.random_process.reset_states()
            done = False
            episode_len = 0
            while not done:
                # env.render()
                if global_timestep.numpy() < agent.params.learning_start:
                    action = env.action_space.sample()
                else:
                    action = agent.predict(state)
                # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
                next_state, reward, done, info = env.step(action * env.action_space.high)
                replay_buffer.add(state, action, reward, next_state, done)

                global_timestep.assign_add(1)
                episode_len += 1
                total_reward += reward
                state = next_state

                # for evaluation purpose
                if global_timestep.numpy() % agent.params.eval_interval == 0:
                    agent.eval_flg = True

            """
            ===== After 1 Episode is Done =====
            """

            # train the model at this point
            for t_train in range(episode_len):  # in mujoco, this will be 1,000 iterations!
                states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
                loss = agent.update(states, actions, rewards, next_states, dones)
                soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.soft_update_tau)
                soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.soft_update_tau)

            tf.contrib.summary.scalar("reward", total_reward, step=i)
            tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
            if i >= agent.params.reward_buffer_ep:
                tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)

            # store the episode reward
            reward_buffer.append(total_reward)
            time_buffer.append(time.time() - start)

            if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss), 0, [0])

            # === Store the trained policy and Scores ===
            if not agent.eval_flg:
                scores.append(total_reward)  # save the reward which has the same amount as `policies` buffer
                weights_vec = flatten_weight(agent.actor.get_weights())
                policies.append(weights_vec)

            # === Train the GP Net ===
            if agent.eval_flg:
                weights_vec = flatten_weight(agent.actor.get_weights())
                gp_model.fit(np.array(policies), np.array(scores))
                sample_ = gp_model.sample_y(weights_vec[np.newaxis, ...], n_samples=num_sample)
                eval_scores = test_Agent_DDPG(agent, env, n_trial=10)
                print("Evaluation Mode => Mean Return: {:.2f} | STD Return: {:.2f} | Mean Est Return: {:.2f} | STD Est Return: {:.2f}"
                      .format(i, eval_scores.mean(), eval_scores.std(), sample_.mean(), sample_.std()))

                # visualisation purpose
                preds.append(sample_)
                evals.append(eval_scores)

                # After training processes
                scores.append(total_reward)
                policies.append(weights_vec)
                agent.eval_flg = False

            # check the stopping condition
            if global_timestep.numpy() > agent.params.num_frames:
                print("=== Training is Done ===")
                gp_model.fit(np.array(policies), np.array(scores))
                sample_ = gp_model.sample_y(weights_vec[np.newaxis, ...], n_samples=num_sample)
                eval_scores = test_Agent_DDPG(agent, env, n_trial=10)
                print("Final Evaluation Mode => Mean Return: {:.2f} | STD Return: {:.2f} | Mean Est Return: {:.2f} | STD Est Return: {:.2f}"
                      .format(i, eval_scores.mean(), eval_scores.std(), sample_.mean(), sample_.std()))

                # visualisation purpose
                preds.append(sample_)
                evals.append(eval_scores)
                test_Agent_DDPG(agent, env, n_trial=agent.params.test_episodes)
                env.close()
                break

        # TODO: Add the visualisation step to plot the est return and actual return line graph at eval phase!!
        preds, evals = np.array(preds)[:, 0, :], np.array(evals)[:, 0, :]
        print(preds.shape, evals.shape)
        plotting_fn(preds, evals)
