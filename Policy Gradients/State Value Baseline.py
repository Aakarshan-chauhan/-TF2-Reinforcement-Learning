import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

import tqdm

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def get_model(input_shape, num_output, hidden_units):
	model = Sequential([
		Dense(hidden_units, activation='tanh', input_shape = input_shape),
		Dense(num_output, activation='linear')
		])

	return model

def get_actions(obs, model):
	logits = model(obs)

	actions = tf.squeeze(tf.random.categorical(logits, 1), axis =1)
	return actions

def get_values(obs,model):
	value = model(obs)
	return value

def get_returns(rewards):
	gamma = 1
	n = len(rewards)
	returns = np.empty_like(rewards)


	for i in reversed(range(n)):
		returns[i] = rewards[i] + gamma * (0 if i+1>=n else returns[i+1])

	return returns


def get_loss(obs, actions, weights, model):
	logits = model(obs)
	values = get_values(obs, value_model)

	action_masks = tf.one_hot(actions, num_actions)
	log_actions = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
	
	with_baseline = weights - values
	
	policy_model_loss = -tf.reduce_mean(log_actions * with_baseline)
	value_model_loss = tf.reduce_mean((values - weights)**2)
	return policy_model_loss, value_model_loss

def play_one_epoch(env, model, value_model):
	experience_len = 1000
	batch_obs = []
	batch_actions=[] 
	batch_rewards = []
	batch_weights = []


	episode_rewards = []

	obs = env.reset()

	steps = 0
	while True:
		steps +=1
		batch_obs.append(obs)
		obs = np.array([obs])

		action = get_actions(obs, model)[0]

		obs, reward, done, info = env.step(action.numpy())
		reward = (steps if done else 1)

		episode_rewards.append(reward)
		batch_actions.append(action)

		if done:
			steps = 0
			batch_rewards.append(np.sum(episode_rewards))
			batch_weights += list(get_returns(episode_rewards))
			
			episode_rewards = []
			obs = env.reset()

			if len(batch_obs) > experience_len:
				break


	
	policy_loss, value_loss = get_loss(np.array(batch_obs), np.array(batch_actions), np.array(batch_weights), model)

	return policy_loss , value_loss, batch_rewards

def train(env, policy_model, value_model):
	optimizer = tf.keras.optimizers.Adam()

	with tf.GradientTape(persistent=True) as tape:
		ploss, vloss, rewards = play_one_epoch(env, policy_model, value_model)

	pgrads = tape.gradient(ploss, policy_model.trainable_variables)
	vgrads = tape.gradient(vloss, value_model.trainable_variables)


	optimizer.apply_gradients(zip(pgrads, policy_model.trainable_variables))
	optimizer.apply_gradients(zip(vgrads, value_model.trainable_variables))

	del(tape)
	return rewards

if __name__ == "__main__":
	env = gym.make("CartPole-v0")
	state = env.observation_space.sample()
	state_shape = np.array([state]).shape
	num_actions = env.action_space.n
	num_epochs=50
	total = []

	policy_model = get_model(state_shape, num_actions, 128)
	value_model = get_model(state_shape, 1, 128)
	for i in tqdm.tqdm(range(num_epochs), desc="training..."):
		rewards = train(env, policy_model, value_model)
		total.append(np.mean(rewards))
		#print(rewards)
	
	plt.xlabel("Epochs")
	plt.ylabel("Mean rewards per epoch")
	plt.plot(total)
	plt.show()
