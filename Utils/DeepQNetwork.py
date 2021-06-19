import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
class DQN(Model):
	def __init__(self, num_actions):
		super(DQN, self).__init__()
		self.dqn = Sequential([
			Dense(1024, activation='relu'),
			Dense(512, activation='relu'),
			Dense(num_actions, activation='softmax')
			])
	
	def get_action(self, state):
		q_vals =self.dqn(state)
		dist=  tfp.distributions.Categorical(probs = q_vals)

		return dist.sample()[0]

	def call(self, state):
		return self.dqn(state)

