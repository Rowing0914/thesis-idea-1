import numpy as np
import tensorflow as tf


def flatten_weight(weights):
    """ Flatten the weight matrix """
    weights_vec = np.array([])
    for weight in weights:
        weights_vec = np.concatenate([weights_vec, weight.flatten()])
    return weights_vec.flatten()


def test_Agent_DDPG(agent, env, n_trial=1):
    """
    Evaluate the trained agent!

    :return:
    """
    all_rewards = list()
    for ep in range(n_trial):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.predict(state)
            # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
            next_state, reward, done, _ = env.step(action * env.action_space.high)
            state = next_state
            episode_reward += reward

        all_rewards.append(episode_reward)
        tf.contrib.summary.scalar("Evaluation Score", episode_reward, step=agent.index_timestep)
    return np.array([all_rewards])


def plotting_fn(preds_mean, preds_std, evals_mean, evals_std):
    """ Plot two sequences with a shaded area representing the corresponding standard deviation"""
    import matplotlib.pyplot as plt
    plt.plot(preds_mean, label="preds mean")
    plt.fill_between(np.arange(len(preds_mean)), preds_mean - 2*preds_std, preds_mean + 2*preds_std, alpha=0.5)
    plt.plot(evals_mean, label="evals mean")
    plt.fill_between(np.arange(len(evals_mean)), evals_mean - 2*evals_std, evals_mean + 2*evals_std, alpha=0.5)
    plt.legend()
    plt.xlabel("Number of Evaluation")
    plt.ylabel("Return")
    plt.show()
