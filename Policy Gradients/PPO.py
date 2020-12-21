import gym
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

class ActorCritic(Model):
	def __init__(self, input_dims, output_dims):
		super(Actor, self).__init__()
		self.d1 = Dense(128, activation='relu')
		self.d2 = Dense(128, activation='relu')
		self.v = Dense(1)
		self.p = Dense(output_dims, activation='softmax')

	def call(self, obs):
		x =self.d1(obs)
		x = self.d2(x)
		return self.p(x), self.v(x)

class ReplayBuffer():
	def __init__(self, size, observation_dims, action_dims):
		self.buffer_size = size
		
		self.obs_mem = np.zeros((self.buffer_size, observation_dims))
		self.next_obs_mem = np.zeros_like(self.obs_mem)
		self.action_mem = np.zeros(self.buffer_size, dtype=np.int64)
		self.reward_mem = np.zeros(self.buffer_size)
		self.terminal_mem = np.zeros(self.buffer_size, dtype=np.float32)

		self.buffer_cntr = 0

	def insert_exp(self, obse, obse_, actione, rewarde, terminale):
		index = int(self.buffer_cntr % self.buffer_size)

		self.obs_mem[index] = obse
		self.next_obs_mem[index] = obse_
		self.action_mem[index] = actione 
		self.reward_mem[index] = rewarde
		self.terminal_mem[index] = 1 - int(terminale)

		self.buffer_cntr +=1

	def get_exp(self, batch_size):
		max_size = int(min(self.buffer_cntr, self.buffer_size))
		batch = np.random.choice(max_size, batch_size)

		batch_obs = self.obs_mem[batch]
		batch_obs_ = self.next_obs_mem[batch]
		batch_action = self.action_mem[batch]
		batch_reward = self.reward_mem[batch]
		batch_terminal = self.terminal_mem[batch]

		return batch_obs, batch_obs_, batch_action, batch_reward, batch_terminal

class Agent()
if __name__ == "__main__":
	env = gym.make("CartPole-v1")

	num_actions = env.action_space.n
	