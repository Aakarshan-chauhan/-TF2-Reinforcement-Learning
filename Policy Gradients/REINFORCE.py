import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def get_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(64, activation='relu'),
		tf.keras.layers.Dense(64, activation='relu'),
		tf.keras.layers.Dense(num_actions, activation = 'softmax')
		])
	return model

def get_action(state):
	state = tf.reshape(state, (-1, len_states))
	logits = model(state)

	action =tf.squeeze(tf.random.categorical(logits, 1))

	action = tf.reshape(action , ())
	return action

def get_returns(rewards, actions):
	length = rewards.shape[0]

	old_g = tf.constant(0.0, dtype=tf.float32)

	G = tf.TensorArray(dtype=tf.float32, size=length, name="Returns")
	for i in reversed(range(length)):
		g = rewards[i] + gamma * old_g
		G = G.write(i, [g])
		old_g = g
	G = G.stack()

	# Normalize
	mean = np.mean(G)
	std = np.std(G) 
	if std <= 0:
		std = 1
	G = (G - mean) / std

	G = tf.keras.backend.one_hot(actions, num_actions) * G
	return G	

def get_score(states, returns):
	states = tf.reshape(states, (-1, len_states))
	logits = model(states)
	loss_pi = tf.reduce_mean(
		returns * tf.math.log(tf.clip_by_value(logits,1e-10,1-1e-10)))

	return -loss_pi



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


def update_model(states, rewards, actions):
	
	with tf.GradientTape() as tape:
		returns = get_returns(rewards,actions)
		logpi = get_score(states,returns)
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

	optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
	gamma = 0.99

	model = get_model()

	num_episodes = 1000
	highreward = float("-inf")
	rews = []
	running_avg = []
	for i in range(num_episodes):
		
		r = train()
		highreward = (r if r > highreward else highreward)
		rews.append(r)
		if i > 100:
			mean = np.mean(rews[i-100:i])
		else:
			mean = np.mean(rews)
		print(f"\rEpisode: {i} / {num_episodes} || Highest reward = {highreward} || current average = {mean}", flush=True, end='')
		
		running_avg.append(mean)
	plt.title("Rewards per episode")
	plt.xlabel("Episodes")
	plt.ylabel("Rewards")
	plt.plot(rews, label="Rewards per episode")
	plt.plot( running_avg, label="Running Average reward")
	plt.legend()
	plt.show()