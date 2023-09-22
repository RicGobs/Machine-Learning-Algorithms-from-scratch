import gymnasium as gym
from collections import deque
from collections import defaultdict
import random
import os
import numpy as np
import matplotlib.pyplot as plt


class table_class:
    def __init__(self, env, learning_rate, epsilon):

        self.env = env
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.size = (env.observation_space.high - env.observation_space.low) / 50 # normalization
        self.q_value_table = np.zeros((50,50,3))

        # state [position speed]
        # self.env.observation_space.low:[-1.2  -0.07]
        # self.env.observation_space.high:[0.6  0.07]
        # self.size:[0.036  0.0028]


    def decresing_epsilon(self):
        if self.epsilon > 0.5 :
          self.epsilon = 90 * self.epsilon / 100
        else :
          self.epsilon = 98 * self.epsilon / 100

        # to have always a bit of randomness
        if self.epsilon < 0.001 :
          self.epsilon = 0.005


    def decresing_learning_rate(self):
        self.learning_rate = 98 * self.learning_rate / 100


    def action(self, state):
        if random.random() > self.epsilon:
            return self.best_action(state[0],state[1])
        else:
             return np.random.choice(3)


    def best_action(self, state_position, state_speed):
        state_discrete_position = ((state_position - self.env.observation_space.low[0]) / self.size[0]).astype(np.int64)
        state_discrete_speed = ((state_speed - self.env.observation_space.low[1]) / self.size[1]).astype(np.int64)
        return np.argmax(self.q_value_table[state_discrete_position][state_discrete_speed]) # take the index of max q value (so the best action) # if all equal, return smallest index


    def discretization(self, state, next_state):
        state_discrete_position = ((state[0] - self.env.observation_space.low[0]) / self.size[0]).astype(np.int64)
        state_discrete_speed = ((state[1] - self.env.observation_space.low[1]) / self.size[1]).astype(np.int64)

        next_state_discrete_position = ((next_state[0] - self.env.observation_space.low[0]) / self.size[0]).astype(np.int64)
        next_state_discrete_speed = ((next_state[1] - self.env.observation_space.low[1]) / self.size[1]).astype(np.int64)

        return state_discrete_position, state_discrete_speed, next_state_discrete_position, next_state_discrete_speed


    def update_table(self, state, next_state, action, reward, terminated):

        state_discrete_position, state_discrete_speed, next_state_discrete_position, next_state_discrete_speed = self.discretization(state, next_state)

        next_best_value = np.max(self.q_value_table[next_state_discrete_position][next_state_discrete_speed]) # take max q_value between the 3 actions

        # 0.95 discount factor
        self.q_value_table[state_discrete_position][state_discrete_speed][action] = ( (1 - self.learning_rate) * self.q_value_table[state_discrete_position][state_discrete_speed][action] + self.learning_rate * ( reward + 0.95 * next_best_value ) )


env = gym.Env
episodes = 1000
average_reward, win_episode = [], 0
learning_rate = 0.01
epsilon = 0.5

env = gym.make('MountainCar-v0', render_mode = "human")
#env = gym.make('MountainCar-v0')

table = table_class(env, learning_rate, epsilon)
table.q_value_table = np.load(f'./table_model/table5.npy')


for episode in range(episodes):

    state, _ = env.reset()
    terminate, truncate, episode_reward = False, False, 0.0

    while not terminate and not truncate:
      action = table.best_action(state[0], state[1])

      next_state, reward, terminate, truncate, _ = env.step(action)
      episode_reward += reward
      state = next_state

      if next_state[0] >= 0.5:
        win_episode += 1

    average_reward.append(episode_reward)

    print(f"Episode {episode}, Reward {episode_reward:.2f}")


mean = sum(average_reward) / len(average_reward)
accuracy = win_episode / episodes

print(f"\n\nAverage Reward: {mean:.2f}, Accuracy {accuracy:.2f}\n")
