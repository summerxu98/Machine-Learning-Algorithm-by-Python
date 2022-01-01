import random
import sys

import numpy as np

from environment import MountainCar

class Error(Exception):
    pass


class LinearModel:
    def __init__(self, state_size: int, action_size: int, lr: float, indices: bool):
        '''
        indices true when one-hot, false when sparse representation
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.bias = 0
        if (indices == True):
            self.weight = np.zeros((2, 3))
        else:
            self.weight = np.zeros((2048, 3))

    def predict(self, state: dict[int, int]) -> list[float]:
        self.state_array = np.zeros((self.state_size, 1))
        # set up bias
        for key in state.keys():
            self.state_array[key, 0] = state[key]
        q_current = np.dot(self.state_array.T, self.weight)
        q_current = q_current + self.bias
        return q_current


    def update(self, state: dict[int, int], action: int, target: float):
        gradient = np.zeros((len(self.weight), len(self.weight[0])))
        state_array = np.zeros((self.state_size, 1))
        for key in state.keys():
            state_array[key, 0] = state[key]
        gradient[:, action] = state_array[:,0]
        #print(gradient)
        self.weight = self.weight - self.lr*target*gradient
        self.bias = self.bias - self.lr*target



class QLearningAgent:
    def __init__(self, env: MountainCar, mode: str = None, gamma: float = 0.9, lr: float = 0.01, epsilon: float = 0.05):
        self.env = env
        self.mode = mode
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.linearmodel = LinearModel(state_size= None, action_size= 3, lr = self.lr, indices=None)
        #dimension of weight = states number * action number
        if (mode == 'raw'):
            self.linearmodel.__init__(2, 3, self.lr, True)
        elif (mode == 'tile'):
            self.linearmodel.__init__(2048, 3, self.lr, False)
        else:
            raise Error("Invalid environment mode. Must be tile or raw")

    def get_action(self, state: dict[int, int]) -> int:
        '''
        epsilon_greedy strategy
        :param state:
        :return: action choice
        '''
        q_current = self.linearmodel.predict(state)
        return np.argmax(q_current[0])

    def train(self, episodes: int, max_iterations: int) -> list[float]:
        '''
        train for episodes, every episodes for max_iterations times
        :param episodes:
        :param max_iterations:
        :return: a list of returns
        '''
        #print(self.env.state)
        res = []
        for i in range(episodes):
            res.append(self.train_episode(max_iterations))
        return res

    def train_episode(self, max_iterations: int) -> int:
        '''
        one episode train
        '''
        returns = 0
        for i in range(max_iterations):
            if(i == 0):
                state = self.env.reset()
            #Take actions and go next step
            random_number = random.random()
            q_current = self.linearmodel.predict(state)
            cur_state = state
            if (random_number < 1 - self.epsilon):
                action = self.get_action(state)
                state, reward, done = self.env.step(action)
            else:
                action = np.random.randint(3)
                state, reward, done = self.env.step(action)
            target = q_current[0][action] - reward - self.gamma * self.cal_maxaction(state)
            returns += reward
            self.linearmodel.update(cur_state, action, target)
            if(done == True):
                break
        return returns

    def cal_maxaction(self, state: dict[int, int]):
        q_current = self.linearmodel.predict(state)
        return np.max(q_current[0])

def main(argv):
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    return_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    lr = float(sys.argv[8])
    env = MountainCar(mode=mode)
    agent = QLearningAgent(env=env, mode=mode, gamma=gamma, epsilon=epsilon, lr=lr)
    returns = agent.train(episodes=episodes, max_iterations=max_iterations)
    return_out = open(return_out, 'wt', encoding='utf-8')
    for r in returns:
        return_out.write(str(r) + '\n')
    weight_out = open(str(weight_out), 'wt', encoding='utf-8')
    weight_out.write(str(agent.linearmodel.bias) + '\n')
    for i in range(len(agent.linearmodel.weight)):
        for j in range(len(agent.linearmodel.weight[i])):
            weight_out.write(str(agent.linearmodel.weight[i][j]) + '\n')

if __name__ == "__main__":
    #main(sys.argv)
    '''
    mode = 'tile'
    env = MountainCar(mode=mode, fixed= 1)
    gamma = 0.99
    epsilon = 0.0
    lr = 0.005
    agent = QLearningAgent(env = env, mode=mode, gamma=gamma, epsilon=epsilon, lr=lr)
    episodes = 25
    max_iterations = 200
    returns = agent.train(episodes=episodes, max_iterations = max_iterations)
    print(agent.linearmodel.bias)
    print(agent.linearmodel.weight)
    print(returns)
    '''


