import gym
import numpy as np
import collections
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tf_rl.common.utils import AnnealingSchedule, eager_setup
from tf_rl.common.wrappers import DiscretisedEnv
from utils.kernels import RBFKernelFn
from utils.gp_models import create_model

eager_setup()

tfd = tfp.distributions


class Q_Agent:
    def __init__(self, env):
        self.env = env
        # (1, 1, 6, 12, 2) => 144 dim vector after being serialised
        self.Q = np.zeros(self.env.buckets + (env.action_space.n,))
        self.gamma = 0.995

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, alpha):
        self.Q[state][action] += alpha * (reward + 1. * np.max(self.Q[next_state]) - self.Q[state][action])

    def test(self):
        """
        Test the agent with a visual aid!
        """

        scores = list()
        for ep in range(10):
            current_state = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.choose_action(current_state, 0)
                obs, reward, done, _ = self.env.step(action)
                current_state = obs
                score += reward
            scores.append(score)
            print("Ep: {}, Score: {}".format(ep, score))
        scores = np.array(scores)
        print("Eval => Std: {}, Mean: {}".format(np.std(scores), np.mean(scores)))


if __name__ == '__main__':
    # DiscretisedEnv
    env = DiscretisedEnv(gym.make('CartPole-v0'))

    # hyperparameters
    num_episodes = 500
    goal_duration = 190
    decay_steps = 5000
    durations = collections.deque(maxlen=10)
    Epsilon = AnnealingSchedule(start=1.0, end=0.01, decay_steps=decay_steps)
    Alpha = AnnealingSchedule(start=1.0, end=0.01, decay_steps=decay_steps)
    agent = Q_Agent(env)

    gp_model = create_model(input_shape=144, kernel_fn=RBFKernelFn)
    batch_size = 50
    num_epochs = 100
    num_sample = 100  # number of sampling

    policies, scores = list(), list()
    _max, _min, _means = list(), list(), list()

    global_timestep = tf.train.get_or_create_global_step()
    # === Data Collection Part ===
    for episode in range(num_episodes):
        current_state = env.reset()

        done = False
        duration = 0

        # one episode of q learning
        while not done:
            duration += 1
            global_timestep.assign_add(1)
            action = agent.choose_action(current_state, Epsilon.get_value())
            new_state, reward, done, _ = env.step(action)
            agent.update(current_state, action, reward, new_state, Alpha.get_value())
            current_state = new_state

        # == After 1 episode ===
        policies.append(agent.Q.flatten())
        scores.append(duration)
        durations.append(duration)

        if episode > batch_size:
            history = gp_model.fit(np.array(policies), np.array(scores), batch_size=batch_size, epochs=num_epochs,
                                   verbose=False)
            sample_ = gp_model(agent.Q.flatten()[np.newaxis, ...]).sample(num_sample).numpy()
            print("Ep: {} | Return: {} | Mean Est Return: {:.2f} | Mean Loss: {:.3f}".format(
                episode, duration, sample_.mean(), np.mean(history.history["loss"]))
            )
            _max.append(sample_.max())
            _min.append(sample_.min())
            _means.append(sample_.mean())

    # check if our policy is good
    # if np.mean(durations) >= goal_duration and episode >= 100:
    # 	print('Ran {} episodes. Solved after {} trials'.format(episode, episode - 100))
    # 	agent.test()
    # 	env.close()
    # 	break

    # === Visualisation Part ===
    _min, _max, _means = np.array(_min), np.array(_max), np.array(_means)
    plt.title("Starting at {} Ep".format(batch_size))
    plt.plot(scores, label="Return")
    plt.plot(_means, label="Est Return")
    plt.fill_between(np.arange(len(_min)), _min, _max, facecolor='blue', alpha=0.5)
    plt.legend()
    plt.show()
