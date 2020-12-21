import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions
import gym
import matplotlib.pyplot as plt

class Policy(Model):
	def __init__(self, input_dims, num_output):
		super(Policy, self).__init__()
		self.d1 = Dense(256, activation='relu', input_shape=(input_dims, ))
		self.d2 = Dense(256, activation='relu')
		self.pi = Dense(num_output, activation='softmax')

	def __call__(self, obs, training=None):
		x = self.d1(obs)
		x = self.d2(x)
		return self.pi(x)

class Agent():
	def __init__(self, alpha, gamma, obs_dims, n_actions):
		self.obs_dims = obs_dims
		self.n_actions = n_actions
		self.gamma = gamma 

		self.model = Policy(self.obs_dims, self.n_actions)
		self.optimizer = Adam(learning_rate=alpha)

		self.statemem = []
		self.actionmem = []
		self.rewardmem = []

	@tf.function
	def get_action(self, obs):
		obs = tf.reshape(obs, (-1, self.obs_dims))
		probs = self.model(obs)
		action_probs = distributions.Categorical(probs=probs)
		action = action_probs.sample()
		return action[0]
		
	def get_returns(self):
		G = np.zeros(len(self.rewardmem))
		g = 0.
		for i in reversed(range(len(self.rewardmem))):
			g += self.gamma * self.rewardmem[i]
			G[i] = g
		return G

	def resetmem(self):
		self.statemem = []
		self.actionmem = []
		self.rewardmem = []

	def train(self):
		states = tf.convert_to_tensor(self.statemem, dtype=tf.float32)
		actions = tf.convert_to_tensor(self.actionmem, dtype=tf.int64)
		returns = tf.convert_to_tensor(self.get_returns(), dtype=tf.float32)


		with tf.GradientTape() as tape:
			probs = self.model(states)
			probs = distributions.Categorical(probs=probs)
			logProbs = probs.log_prob(actions)
			policyLoss = -tf.reduce_sum(logProbs*returns)
		policyGrads = tape.gradient(policyLoss, self.model.trainable_variables) 
		self.optimizer.apply_gradients(zip(policyGrads, self.model.trainable_variables))


if __name__ == "__main__":
	env = gym.make("LunarLander-v2")
	obs_dims = env.reset().shape[0]
	n_actions = env.action_space.n
	gamma = 0.999
	alpha = 0.00001
	
	agent = Agent(alpha, gamma, obs_dims, n_actions)

	scores = []
	avg_score = []
	for i in range(2001):
		s = env.reset()
		done = False
		
		while not done:
			a = agent.get_action(s).numpy()
			s_, r, done , info = env.step(a)
			agent.actionmem.append(a)
			agent.statemem.append(s)
			agent.rewardmem.append(r)
			s = s_
			
		agent.train()
		scores.append(np.sum(agent.rewardmem))
		avg_score.append(np.mean(scores[-50:]))
		if i%50 == 0 :
			print(f"Episode : {i}, avg_reward = {avg_score[-1]}")
		agent.resetmem()


	plt.plot(avg_score)
	plt.show()