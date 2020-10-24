import tensorflow as tf
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import tqdm
def get_network(input_shape, output_shape, hidden_units):
	model = Sequential([
		Dense(13, activation='tanh', input_shape=input_shape),
		Dense(output_shape, activation='softmax')
		])

	return model


def play_one(env:gym.envs, model:tf.keras.Model):
	done = False
	obs = env.reset()
	episode_obs = []
	episode_actions =[]
	episode_rewards =[]

	i = 0
	while not done:
		episode_obs.append(obs)
		obs = tf.constant([obs], dtype=tf.float32)
		
		action_logits = model(obs)
		action  = tf.random.categorical(action_logits, 1)[0, 0]

		episode_actions.append(action)

		obs, reward, done, _ = env.step(action.numpy())
		episode_rewards.append(reward)

		if done:
			
			break
		i+=1

	return episode_actions, episode_rewards, episode_obs

def get_returns(rewards, gamma=1):
	returns = []

	ret = tf.constant(0.0)
	for i in tf.range(len(rewards)):
		ret = tf.reduce_sum(rewards)
		returns.append(ret)

	return returns

def get_loss(actions, weights, obs, model):
	action_probs = model(obs)

	actions = tf.squeeze(tf.random.categorical(action_probs,1), axis=1)


	action_masks = tf.one_hot(actions, num_actions)

	log_actions = tf.reduce_sum(action_masks * tf.math.log(action_probs), axis = 1)

	loss = -1*tf.reduce_mean(log_actions * weights)
	

	return loss


def train(env, gamma, model, optimizer, batch_size=1000):

	batch_returns = []
	batch_obs = []
	batch_acts = []
	batch_weights = []
	batch_lens = []
	

	while len(batch_obs) < batch_size:
		actions , rewards, obs = play_one(env, model)
		returns = get_returns(rewards, gamma)

		batch_obs += obs
		batch_lens.append(len(returns))
		batch_acts.append(actions)
		returns = np.sum(rewards)

		batch_returns.append(returns)
		for i in range(len(rewards)):

			batch_weights.append([returns])
		
	
	batch_obs = np.array(batch_obs)
	batch_weights = np.array(batch_weights)
	batch_weights = np.reshape(batch_weights, newshape=(-1, 1))


	batch_weights = tf.cast(batch_weights, tf.float32)
	with tf.GradientTape() as tape:
		losses = get_loss(batch_acts, batch_weights, [batch_obs], model)
	policy_grads = tape.gradient(losses, model.trainable_variables)
	optimizer.apply_gradients(zip(policy_grads, model.trainable_variables))
	return batch_returns


if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	num_actions = env.action_space.n
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
	sample_obs= tf.constant( [env.reset()], tf.float32)
	obs_shape = sample_obs.shape

	network = get_network(obs_shape, num_actions, 128)

	total = []
	for i in tqdm.tqdm(range(1000)):
		rewards = train(env, 1, network, optimizer)
		
		total.append(np.mean(rewards))
	

	plt.xlabel("Epochs")
	plt.ylabel("total reward per episode")
	plt.plot(total)
	plt.show()