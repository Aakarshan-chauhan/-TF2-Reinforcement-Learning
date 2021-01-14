import tensorflow as tf
import tqdm
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
import random
import gym
import wandb


class ReplayBuffer:
	def __init__(self, max_size, observation_dims, n_actions):
		self.max_size = max_size
		self.obervation_dims = observation_dims
		self.n_actions = n_actions

		self.obs_mem = np.zeros([self.max_size, observation_dims], dtype=np.float32)
		self.next_obs_mem = np.zeros_like(self.obs_mem)

		self.action_mem = np.zeros(self.max_size, dtype=np.int64)
		self.reward_mem = np.zeros(self.max_size, dtype=np.float32)

		self.done_mem = np.zeros(self.max_size, dtype=np.float32)
		self.mem_cntr = 0

	def store(self, s, a, r, s_, d):
		index = self.mem_cntr % self.max_size

		self.obs_mem[index] = s
		self.next_obs_mem[index] = s_
		self.action_mem[index] = a
		self.reward_mem[index] = r
		self.done_mem[index] = d
		self.mem_cntr += 1

	def sample(self, batch_size):
		max_batch = min(self.mem_cntr, self.max_size)
		batches = np.random.choice(max_batch, batch_size)

		s = self.obs_mem[batches]
		a = self.action_mem[batches]
		r = self.reward_mem[batches]
		s_ = self.next_obs_mem[batches]
		d = self.done_mem[batches]
		return s, a, r, s_, d


class QNetwork(Model):
	def __init__(self, n_actions):
		super(QNetwork, self).__init__()
		self.d1 = Dense(256, activation='relu')
		self.d2 = Dense(256, activation='relu')
		self.q = Dense(n_actions)

	def __call__(self, s, training=None):
		x = self.d1(s)
		x = self.d2(x)
		x = self.q(x)
		return x


class Agent():
	def __init__(self, obs_dims, n_actions, buffer_size, batch_size, alpha, gamma, epsilon, tau):
		self.obs_dims = obs_dims
		self.n_actions = n_actions
		self.batch_size = batch_size
		self.gamma = gamma
		self.e = epsilon
		self.tau = tau

		self.buffer = ReplayBuffer(buffer_size, obs_dims, n_actions)
		self.Q = QNetwork(n_actions)
		self.targetQ = QNetwork(n_actions)
		self.Q.compile(optimizer=Adam(learning_rate=alpha), loss='mse')

		self.opt = Adam(learning_rate=alpha)
		self.update_target_parameters()

	# @tf.function
	def get_action(self, s):
		if random.random() < self.e:
			return np.random.randint(high=self.n_actions, low=0)

		s = np.reshape(s, (-1, self.obs_dims))
		q = self.Q.predict(s)
		return np.argmax(q)

	# @tf.function
	def learn(self):
		s, a, r, s_, d = self.buffer.sample(self.batch_size)

		self.e = self.e * 0.994

		Q = self.Q.predict(s)
		Q_next = self.targetQ.predict(s_)

		targets = Q.copy()
		indices = np.arange(self.batch_size, dtype=np.int64)

		targets[indices, a] = r + self.gamma * np.max(Q_next, axis=1) * (1 - d)

		_ = self.Q.fit(s, targets, verbose=0)

	# self.update_target_parameters(tau=self.tau)
	# return tf.reduce_mean(MSE(self.Q.predict(s), Q_next))

	# @tf.function
	def update_target_parameters(self, tau=1.):
		self.targetQ.set_weights(self.Q.get_weights())


if __name__ == "__main__":
	tf.compat.v1.disable_eager_execution()
	env = gym.make("CartPole-v0")
	wandb.init(project='DQN CartPole')

	OBS_DIMS = env.reset().shape[0]
	N_ACTIONS = env.action_space.n
	BUFFER_SIZE = 1000000
	BATCH_SIZE = 128
	LEARNING_RATE = 0.001
	GAMMA = 0.99
	EPSILON = .8
	TAU = 0.01
	agent = Agent(OBS_DIMS, N_ACTIONS, BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE, GAMMA, EPSILON, TAU)

	wandb.config.BUFFER_SIZE = BUFFER_SIZE
	wandb.config.BATCH_SIZE = BATCH_SIZE
	wandb.config.LEARNING_RATE = LEARNING_RATE
	wandb.config.GAMMA = GAMMA
	wandb.config.EPSILON = EPSILON
	wandb.config.TAU = TAU
	n_games = 1000
	episode_rewards = []
	n_steps = 0
	L = 0
	timesteps = 0
	for i in tqdm.trange(n_games):

		obs = env.reset()
		done = False
		episode_reward = 0.
		while True:
			action = agent.get_action(obs)
			obs_, reward, done, info = env.step(action)
			agent.buffer.store(obs, action, reward, obs_, done)
			obs = obs_

			episode_reward += reward

			if done:
				done = False
				obs = env.reset()
				episode_rewards.append(episode_reward)

				avg_reward = np.mean(episode_rewards[-100:])
				wandb.log({'Episode Reward': episode_reward, 'avg_rewards': avg_reward, 'Q Loss': L, 'eps': agent.e})
				episode_reward = 0
				if agent.buffer.mem_cntr > BATCH_SIZE:
					L = agent.learn()
					if timesteps % 15 == 0:
						agent.update_target_parameters()
					break
			timesteps += 1
