import gym
import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions


class Actor(Model):
	def __init__(self, input_dims, num_outputs):
		super(Actor, self).__init__()
		self.d1 = Dense(1024, activation='relu', input_shape=(input_dims, ))
		self.d2 = Dense(512, activation='relu')
		self.pi = Dense(num_outputs, activation='softmax')

	def __call__(self, obs, training=None):
		x = self.d1(obs)
		x = self.d2(x)
		return self.pi(x)

class Critic(Model):
	def __init__(self, input_dims):
		super(Critic, self).__init__()
		self.d1 = Dense(1024, activation='relu', input_shape=(input_dims, ))
		self.d2 = Dense(512, activation='relu')
		self.value = Dense(1, activation='linear')
	
	def __call__(self, obs):
		x = self.d1(obs)
		x = self.d2(x)
		return self.value(x)

class Agent():
	def __init__(self, alpha, gamma, obs_dims, n_actions):
		self.alpha = alpha
		self.gamma = gamma
		self.obs_dims = obs_dims
		self.n_actions = n_actions

		self.statemem = []
		self.rewardmem = []
		self.actionmem = []

		self.actor = Actor(self.obs_dims, self.n_actions)
		self.critic = Critic(self.obs_dims)

		self.opt = Adam(learning_rate = self.alpha)

	def push(self, s, r, a):
		self.statemem.append(s)
		self.rewardmem.append(r)
		self.actionmem.append(a)

	@tf.function
	def get_action(self, s):
		s = tf.reshape(s, (-1, self.obs_dims))
		probs = self.actor(s)
		probs = distributions.Categorical(probs=probs)
		a = probs.sample()
		return a[0]

	def get_returns(self):
		returns = np.zeros(len(self.rewardmem))
		G = 0.

		for i in reversed(range(len(self.rewardmem))):
			G = self.rewardmem[i] + self.gamma * G
			returns[i] = G
		return returns

	def reset(self):
		self.statemem = []
		self.rewardmem = []
		self.actionmem = []

	def learn(self):
		states = tf.reshape(self.statemem, (-1, self.obs_dims))
		rewards = tf.reshape(self.rewardmem, (-1))
		actions = tf.reshape(self.actionmem, (-1))
		returns = self.get_returns()

		

		with tf.GradientTape(persistent=True) as tape:
			pi = self.actor(states)

			probs = distributions.Categorical(probs = pi)
			logProbs = probs.log_prob(actions)

			value = tf.squeeze(self.critic(states))
			advantage = returns - value
			
			critic_loss = tf.reduce_mean(advantage**2)
			actor_loss = -tf.reduce_mean(logProbs * (advantage))

		policyGrads = tape.gradient(actor_loss, self.actor.trainable_variables)
		criticGrads = tape.gradient(critic_loss, self.critic.trainable_variables)

	
		

		self.opt.apply_gradients(zip(policyGrads, self.actor.trainable_variables))
		self.opt.apply_gradients(zip(criticGrads, self.critic.trainable_variables))
		del tape

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
				agent.push(obs, reward, action)
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