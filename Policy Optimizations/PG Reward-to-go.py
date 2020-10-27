import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tqdm


def get_network(inputshape, n_actions, hidden):
	m = Sequential([
		Dense(hidden, activation='tanh', input_shape=inputshape),
		Dense(n_actions , activation='linear')
		])
	return m

def reward_to_go(rewards):
	rtg = np.zeros_like(rewards)
	r= 0
	n = len(rewards)
	for i in reversed(range(n)):
		rtg[i] = rewards[i] + (rtg[i+1] if i+1<n else 0)
	return rtg

@tf.function
def getActions(model, obs):
	logits = model([obs])
	actions = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
	return actions


def getLoss(obs, actions, weights, model ):
	#print(obs)
	action_logits = model(obs)
	#print(action_logits)
	action_masks = tf.one_hot(actions, num_actions)
	#print(action_masks)
	log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(action_logits), axis=1)
	#print(log_probs)
	#print(weights)
	return -tf.reduce_mean(weights * log_probs)


def play_one_epoch(env, model, experiance_len=5000, render = False):
	batch_returns= []
	batch_actions= []
	batch_obs = []
	batch_rewards = []
	batch_lens = []
	episode_rewards = []
	batch_weights= []
	obs = env.reset()
	
	steps = 0
	while True:
		steps+=1
		batch_obs.append(obs)

		action = getActions(model, obs.reshape(1,-1))[0]
		obs, reward, done, info = env.step(action.numpy())

		reward = (steps if done else 1)
		batch_actions.append(action)
		episode_rewards.append(reward)

		if done:
			steps=0
			ep_ret, ep_len = sum(episode_rewards), len(episode_rewards)
			batch_returns.append(ep_ret)
			batch_lens.append(ep_len)

			batch_weights += list(reward_to_go(episode_rewards))
			#print(f"REWARD TO GO : {reward_to_go(episode_rewards)}")
			obs = env.reset()
			done = False
			episode_rewards = []
			#print(f"BATCH OBS LEN : {len(batch_obs)}")
			if len(batch_obs) > experiance_len:
				break

	batch_loss = getLoss(np.array(batch_obs), np.array(batch_actions), np.array(batch_weights), model)
	return batch_loss, batch_returns

def train(env, model):
	optimizer = tf.keras.optimizers.Adam()
	with tf.GradientTape() as tape:
		losses,returns = play_one_epoch(env, model)

	grads = tape.gradient(losses, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return returns
if __name__ == "__main__":
	env = gym.make("CartPole-v0")
	num_actions = env.action_space.n
	sample_state = env.reset()
	state_shape = tf.constant([sample_state]).shape
	rets = []
	model = get_network(state_shape, num_actions, 128)
	for i in tqdm.tqdm(range(50)):
		rets.append(tf.reduce_mean(train(env, model)))

	plt.xlabel("epochs")
	plt.ylabel("mean rewards per epoch")
	plt.plot(rets)
	plt.show()

