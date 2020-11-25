import numpy as np
import matplotlib.pyplot as plt
import tqdm
import tensorflow as tf
import gym

def get_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(num_actions, activation='linear')
		])
	return model

def get_action(state):
	state = np.expand_dims(state, 0).astype(np.float32)
	Q = model(state)
	if np.random.random() < eps:
		return env.action_space.sample()
	return tf.argmax(Q, axis=1)[0].numpy()

def get_returns(rewards, states):
	old_g = 0.
	G = np.empty((len(states)))

	for i in reversed(range(len(rewards))):
		G[i] = rewards[i] + gamma*old_g
		old_g = G[i]

	return G

def updatemodel(states, actions, returns):

	with tf.GradientTape() as tape:
		preds = model(tf.reshape(states, (-1,len_states)))
		action_batch = [[i, actions[i]] for i in range(len(actions))]
		pred = tf.gather_nd(preds, action_batch)
		error = losses(returns, pred)
	grads = tape.gradient(error, model.trainable_variables)

	optimizer.apply_gradients(zip(grads, model.trainable_variables))

def play():
	states = []
	actions = []
	rewards = []

	done = False
	s = env.reset()
	states.append(s)
	rewards.append(0)
	stepcount = 0
	while not done:
		stepcount+=1
		a = get_action(s)
		actions.append(a)

		s,r,done,info = env.step(a)

		states.append(s)
		rewards.append(r)
	
	actions.append(a)


	returns = get_returns(rewards, states)
	returns = tf.expand_dims(returns, 1)
	updatemodel(states, actions , returns)
	return tf.reduce_sum(rewards)

if __name__ == "__main__":
	env = gym.make("CartPole-v0")
	gamma = 0.99
	optimizer = tf.keras.optimizers.Adam()
	len_states= len(env.reset())
	num_actions = env.action_space.n
	losses = tf.keras.losses.Huber()
	model = get_model()
	rews= []

	
	for i in tqdm.trange(500):
		eps = 1 / np.sqrt(i+1)
		rews.append(play())

	avg_10_rewards = [np.mean(rews[i:i+10]) for i in range(len(rews)- 10)]
	plt.xlabel('Episodes')
	plt.ylabel('Rewards')

	plt.plot(avg_10_rewards)

	plt.show()
		