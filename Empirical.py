import numpy as np

from q_learning import QLearningAgent
from environment import MountainCar
import matplotlib.pyplot as plt

def draw_plot(mode, episodes, max_iterations, epsilon, gamma, lr):
    env = MountainCar(mode=mode, fixed= 0)
    agent = QLearningAgent(env=env, mode=mode, gamma=gamma, epsilon=epsilon, lr=lr)
    return agent.train(episodes=episodes, max_iterations = max_iterations)

if __name__ == "__main__":
    raw_episodes = 400
    max_iterations = 200
    epsilon = 0.05
    gamma = 0.99
    lr = 0.00005
    returns_raw = draw_plot('tile', raw_episodes, max_iterations, epsilon, gamma, lr)
    print(returns_raw)
    episode_plot = [i+1 for i in range(raw_episodes)]
    returns_raw = np.array(returns_raw)


    def moving_average(a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    rolling = moving_average(returns_raw, 25)
    rolling = np.append(returns_raw[0:24], rolling)
    print(len(rolling))
    plt.plot(episode_plot, returns_raw, label = 'returns')
    plt.plot(episode_plot, rolling, label = 'rolling mean')
    plt.legend()
    plt.show()