import gymnasium as gym
import argparse
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense

class nnq:
    def __init__(self, env, learning_rate, epsilon):

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.net = self.network()

    def network(self):

        model = Sequential() #input_shape = 2 -> position and speed

        model.add(Dense(24, input_shape=env.observation_space.shape, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(36, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',kernel_initializer='he_uniform'))
        model.add(Dense(3, activation='linear',kernel_initializer='he_uniform'))

        opt = tf.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, decay_steps=300, decay_rate=0.96, staircase=True))
        model.compile(optimizer=opt, loss='mse', metrics=["mse"])

        return model


    def best_action(self,state):
        best_action_to_do = self.net(np.array([state]))
        return np.argmax(best_action_to_do[0], axis=0)


    def action(self, state):
        if random.random() > self.epsilon :
            return self.best_action(state)
        else:
             return np.random.choice(3)

    def replay_buffer(self, buffer, batch):

        buffer_batch = random.sample(buffer, batch)

        state = np.array([i[0] for i in buffer_batch])
        action = np.array([i[1] for i in buffer_batch])
        next_state = np.array([i[2] for i in buffer_batch])
        reward = np.array([i[3] for i in buffer_batch])
        terminate = np.array([i[4] for i in buffer_batch])

        current = self.net(state) #actual rewards
        #target = np.zeros((state.shape[0],3))
        target = np.copy(current)

        next_q = self.net(next_state) #future rewards
        max_next_q = np.amax(next_q, axis=1) #max reward in future reward (after three actions)

        for i in range(state.shape[0]):
            target[i][action[i]] = reward[i] + 0.99 * (1 - terminate[i]) * max_next_q[i]

        self.net.fit(x=state, y=target, epochs=1,verbose=0)
        self.learning = self.net.optimizer.learning_rate.numpy()


    def decresing_epsilon(self):
        if self.epsilon > 0.3 :
          self.epsilon = 95 * self.epsilon / 100
        else :
          self.epsilon = 995 * self.epsilon / 1000

        # to have always a bit of randomness
        if self.epsilon < 0.001 :
          self.epsilon = 0.009


env = gym.make('MountainCar-v0', render_mode = "human")
#env = gym.make('MountainCar-v0')

average_reward, win_episode = [], 0
learning_rate=0.01
epsilon= 1.0
episodes = 100

nnq_class = nnq(env, learning_rate, epsilon)
nnq_class.net = load_model(f'./nn_model/nn7.h5')


for episode in range(episodes):

    state, _ = env.reset()
    terminate, truncate, episode_reward = False, False, 0.0

    while not terminate and not truncate:
      action = nnq_class.best_action(state)

      next_state, reward, terminate, truncate, _ = env.step(action)
      episode_reward += reward
      state = next_state

      if next_state[0] >= 0.5:
        win_episode += 1

    average_reward.append(episode_reward)
    print(f"Episode: {episode} - Episode reward: {episode_reward:.2f}")


mean = sum(average_reward) / len(average_reward)
accuracy = win_episode / episodes

print(f"\n\nAverage Reward: {mean:.2f}, Accuracy {accuracy:.2f}\n")
