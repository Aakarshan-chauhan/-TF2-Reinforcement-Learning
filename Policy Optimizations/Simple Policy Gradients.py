import tensorflow as tf
import numpy as np
import gym
import tqdm

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

def get_model(input_shape, output_num, hidden_units):
	model = Sequential([
		Dense(hidden_units, activation='tanh', input_shape=input_shape),
		Dense(output_num, activation='linear')
		])
	return model


def get_actions(obs, model):
	logits = model(obs)
	return tf.squeeze(tf.random.categorical(logits, 1), axis=1)

def get_returns(rewards):
	n = len(rewards)

	returns = np.empty_like(rewards)
	for i in range(n):
		returns[i] = np.sum(rewards)

	return returns

def get_loss(obs, actions, weights, model):
	logits = model(obs)

	action_masks = tf.one_hot(actions, num_actions)
	log_p = tf.reduce_sum(action_masks*tf.nn.log_softmax(logits), axis=1)
	weights = tf.cast(weights, tf.float32)
	loss = -tf.reduce_mean(log_p * weights)
	return loss

def play_one_epoch(env,model):

	exp_size=5000

	batch_weights = []
	batch_rewards = []
	batch_obs = []
	batch_actions = []
	episode_rewards = []

	obs = env.reset()
	while True:
		batch_obs.append(obs)
		action = get_actions(tf.reshape(obs, (1,-1)), model)[0]

		obs, _, done, info = env.step(action.numpy())
		
		reward = (-1 if done else 1)
		batch_actions.append(action)
		episode_rewards.append(reward)


		if done:

			batch_rewards.append(np.sum(episode_rewards))

			batch_weights += list(get_returns(episode_rewards))

			obs = env.reset()
			episode_rewards = []
			
			if len(batch_obs) > exp_size:
				break

	
	losses = get_loss(np.array(batch_obs), np.array(batch_actions), np.array(batch_weights), model)
	return losses, batch_rewards

def train(env, model):
	
	with tf.GradientTape() as tape:
		losses,rewards = play_one_epoch(env, model)

	grads = tape.gradient(losses, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return rewards


if __name__ == "__main__":
	env = gym.make("CartPole-v0")
	num_actions = env.action_space.n
	sample_state = np.array([env.reset()])

	optimizer = tf.keras.optimizers.Adam()

	model = get_model(sample_state.shape, num_actions, 128)
	num_epochs = 50
	epoch_rewards = []


	for i in tqdm.tqdm(range(num_epochs)):
		rewards = train(env, model)
		epoch_rewards.append(np.mean(rewards))


	plt.xlabel('Epochs')
	plt.ylabel('avg reward per epoch')
	plt.plot(epoch_rewards)
	plt.show()
	print("done")