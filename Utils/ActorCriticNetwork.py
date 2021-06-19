import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, Sequential
import tensorflow_probability as tfp
import gym

class Discrete(Model):
	def __init__(self, input_shape, num_actions):
		
		super(Discrete, self).__init__()
		self.actor = Sequential([
				Dense(1024, activation='relu'),
				Dense(128, activation='relu'),
				Dense(num_actions, activation='softmax')
			])

		self.critic = Sequential([
				Dense(1024, activation='relu'),
				Dense(128, activation='relu'),
				Dense(1)
			])


	def get_action(self, observations):
		pi = self.actor(observations)
		action_probs = tfp.distributions.Categorical(probs=pi)

		return action_probs

	
	def get_value(self, observations):
		value = self.critic(observations)
		return value
	

class Continuous(Model):
	def __init__(self, input_shape, num_actions):
		
		super(Continuous, self).__init__()
		self.actor = Sequential([
				Dense(1024, activation='relu'),
				Dense(128, activation='relu'),
				Dense(num_actions, activation='tanh')
			])

		self.critic = Sequential([
				Dense(1024, activation='relu'),
				Dense(128, activation='relu'),
				Dense(1)
			])


	def get_action(self, observations, variance = 0):
		mu = self.actor(observations)
		action_probs = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_identity_multiplier=variance)
		return action_probs

	
	def get_value(self, observations):
		value = self.critic(observations)
		return value

