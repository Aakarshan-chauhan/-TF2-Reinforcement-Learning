import gym
import numpy as np
import tensorflow as tf
import tqdm
import matplotlib.pyplot as plt
def get_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(32, activation = 'tanh'),
		tf.keras.layers.Dense(32, activation='tanh'),
		tf.keras.layers.Dense(num_actions, activation = 'linear')
		])
	return model


def get_action(state):
	state = tf.reshape(state, (-1, len_states))
	logits = model(state)

	action =tf.squeeze(tf.random.categorical(logits, 1))

	action = tf.reshape(action , ())
	return action


def get_returns(rewards):
	length = rewards.shape[0]

	old_g = tf.constant(0.0, dtype=tf.float32)

	G = tf.TensorArray(dtype=tf.float32, size=length, name="Returns")
	for i in reversed(range(length)):
		g = rewards[i] + gamma * old_g
		G = G.write(i, [g])
		old_g = g
	G = G.stack()
	return G	

def play_episode():
	states = []
	actions = []
	rewards = []
	done = False

	s = env.reset()
	stepcount = 0
	while not done:
		a = get_action(s).numpy()
		stepcount +=1
		actions.append(a)
		states.append(s)
		s,r,done,info = env.step(a)
		
		rewards.append(r)

	rewards = np.array(rewards, dtype=np.float32)
	states = np.array(states, dtype=np.float32)
	actions = np.array(actions , dtype=np.int64)
	return states, rewards, actions

def get_score(states, actions, returns):
	states = tf.reshape(states, (-1, len_states))
	logits = model(states)
	loss_pi = tf.reduce_mean(
		returns * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions))
	return -loss_pi


def update_model(states, rewards, actions):
	
	with tf.GradientTape() as tape:
		returns = get_returns(rewards)
		logpi = get_score(states, actions,returns)
	grad = tape.gradient(logpi, model.trainable_variables)
	optimizer.apply_gradients(zip(grad, model.trainable_variables))

def train():
	states, rewards, actions = play_episode()
	update_model(states, rewards, actions)
	return np.sum(rewards)

if __name__ == "__main__":
	env = gym.make("CartPole-v1")
	len_states = len(env.reset())
	num_actions = env.action_space.n

	optimizer = tf.keras.optimizers.Adam()
	gamma = 0.99

	rews = []
	rs = []
	model = get_model()
	s = env.reset()
	for i in range(500):
		
		rews.append(train())
		if i %10 == 0:
			print(f"Episode: {i}, reward = {np.mean(rews)}")
			rs.append(np.mean(rews))
			rews = []
	plt.plot(rs)
	plt.show()