import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tqdm
import gym
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions

class PolicyNetwork(Model):
	def __init__(self, input_dims, num_outputs):
		super(PolicyNetwork, self).__init__()
		self.d1 = Dense(1024, activation='relu', input_shape=(input_dims, ))
		self.d2 = Dense(512, activation='relu')
		self.pi = Dense(num_outputs, activation='softmax')

	def __call__(self, obs, training=None):
		x = self.d1(obs)
		x = self.d2(x)
		return self.pi(x)


class Agent():
	def __init__(self, alpha, gamma, obs_dims, n_actions):
		self.alpha = alpha
		self.gamma = gamma

		self.obs_dims = obs_dims
		self.n_actions = n_actions

		self.model = PolicyNetwork(self.obs_dims, self.n_actions)

		self.opt = Adam(learning_rate=self.alpha)

		self.statemem = []
		self.actionmem = []
		self.rewardmem = []

	def push(self, s, a, r):
		self.statemem.append(s)
		self.actionmem.append(a)
		self.rewardmem.append(r)

	def reset(self):
		self.statemem = []
		self.actionmem = []
		self.rewardmem = []


	def get_returns(self):
		returns = np.zeros(len(self.rewardmem))
		g = 0.
		for i in reversed(range(len(self.rewardmem))):
			g = self.rewardmem[i] + self.gamma * g
			returns[i] = g

		return returns
	@tf.function
	def get_action(self, obs):
		obs = tf.reshape(obs, (-1, self.obs_dims))
		probs = self.model(obs)
		action_probs = distributions.Categorical(probs=probs)
		action = action_probs.sample()[0]
		return action

	def learn(self):
		states = tf.convert_to_tensor(self.statemem, dtype=tf.float32)
		actions = tf.convert_to_tensor(self.actionmem, dtype=tf.int64)
		returns = self.get_returns()
		
		with tf.GradientTape() as tape:
			probs = self.model(states)
			probs = distributions.Categorical(probs=probs)
			
			logprobs = probs.log_prob(actions)
			policyLoss = -tf.reduce_sum(logprobs * returns)

		policyGrads = tape.gradient(policyLoss, self.model.trainable_variables)
		self.opt.apply_gradients(zip(policyGrads, self.model.trainable_variables))
		self.reset()

if __name__ == '__main__':
	env = gym.make("CartPole-v0")
	obs_dims = env.reset().shape[0]
	n_actions = env.action_space.n

	agent = Agent(0.0005, 0.99, obs_dims, n_actions)

	avg_rewards = []
	scores = []
	with tqdm.trange(300) as t:
		for episode in t:

			obs = env.reset()
			done = False
			score = 0

			while not done:
				action = agent.get_action(obs).numpy()
				obs_, reward, done, info = env.step(action)
				agent.push(obs, action, reward)
				obs = obs_

				score += reward

			agent.learn()
			scores.append(score)

			avg_rewards.append(np.mean(scores[-10:]))
			t.set_postfix(
				avg_reward = avg_rewards[-1]
				)
	plt.xlabel("Episodes")
	plt.ylabel("Avgerage rewards over 10 episodes")

	plt.plot(avg_rewards)
	plt.show()
