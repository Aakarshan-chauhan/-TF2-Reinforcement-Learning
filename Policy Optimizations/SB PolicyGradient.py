import tensorflow as tf
import numpy as np
import gym

def get_network(input_shape, output_num, hidden_units):
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(hidden_units, input_shape = input_shape, activation='tanh'),
		tf.keras.layers.Dense(output_num, activation='linear')
		])
	return model


if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	sample_state = np.array(env.reset())
	num_actions = env.action_space.n
	hidden_units = 128

	gamma = 1

	model = get_network(sample_state.shape, num_actions, hidden_units)
	