import tensorflow as tf
import gym
import numpy 
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

	episode_obs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	episode_actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	episode_rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

	i = 0
	while not done:
		episode_obs = episode_obs.write(i, obs)
		obs = tf.constant([obs], dtype=tf.float32)
		
		action_logits = model(obs)
		action  = tf.random.categorical(action_logits, 1)[0, 0]
		episode_actions = episode_actions.write(i, action)


		obs, reward, done, _ = env.step(action.numpy())
		episode_rewards = episode_rewards.write(i, reward)

		if done:
			
			break
		i+=1

	episode_obs = episode_obs.stack()
	episode_actions = episode_actions.stack()
	episode_rewards = episode_rewards.stack()
	return episode_actions, episode_rewards, episode_obs

def get_returns(rewards, gamma=1):
	returns = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	
	ret = tf.constant(0.0)
	for i in tf.range(rewards.shape[0]):
		ret = tf.reduce_sum(rewards)
		returns = returns.write(i, ret)

	returns = returns.stack()
	return returns

def get_loss(actions, returns, obs):
	action_probs = model(obs)
	log_actions = tf.reduce_sum(tf.math.log(actions))
	loss = -1*tf.reduce_mean(log_actions * returns)
	return loss


def train(env, gamma, model, optimizer):

	batch_returns = []
	batch_obs = []
	epoch_reward = 0.

	while len(batch_returns) < batch_size:
		actions , rewards = play_one(env, model)
		returns = get_returns(rewards, gamma)
		batch_returns += [returns]*len(rewards)

	with tf.GradientTape() as tape:
		losses = get_loss(actions, returns)

	policy_grads = tape.gradient(losses, model.trainable_variables)
	optimizer.apply_gradients(zip(policy_grads, model.trainable_variables))
	return tf.reduce_sum(rewards)


if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	num_actions = env.action_space.n
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
	sample_obs= tf.constant( [env.reset()], tf.float32)
	obs_shape = sample_obs.shape

	network = get_network(obs_shape, num_actions, 128)

	total = []
	for i in tqdm.tqdm(range(5000)):
		rewards = train(env, 1, network, optimizer)
		
		total.append(rewards)
	

	plt.xlabel("Episodes")
	plt.ylabel("total reward per episode")
	plt.plot(total)
	plt.show()