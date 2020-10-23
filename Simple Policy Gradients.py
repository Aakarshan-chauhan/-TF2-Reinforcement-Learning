import tensorflow as tf
import gym
import numpy 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def get_network(input_shape, output_shape, hidden_units):
	model = Sequential([
		Dense(hidden_units, activation='relu', input_shape=input_shape),
		Dense(output_shape, activation='linear')
		])

	return model


def play_one(env:gym.envs, model:tf.keras.Model):
	done = False
	obs = env.reset()

	episode_actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	episode_rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

	i = 0
	while not done:
		obs = tf.constant([obs], dtype=tf.float32)

		action_logits = model(obs)
		action  = tf.random.categorical(action_logits, 1)[0, 0]
		action_logits = tf.nn.log_softmax(action_logits)
		episode_actions = episode_actions.write(i, action_logits[0,action])


		obs, reward, done, _ = env.step(action.numpy())
		episode_rewards = episode_rewards.write(i, reward)

		if done:
			
			break
		i+=1
	episode_actions = episode_actions.stack()
	episode_rewards = episode_rewards.stack()
	return episode_actions, episode_rewards 

def get_returns(rewards, gamma=1):
	returns = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
	
	ret = tf.constant(0.0)
	for i in tf.range(rewards.shape[0]):
		ret += gamma*rewards[i]
		returns = returns.write(i, ret)

	returns = returns.stack()
	return returns

def get_loss(actions, returns):
	log_a = tf.math.log(actions)
	loss = tf.reduce_mean(log_a * returns)

if __name__ == "__main__":
	env = gym.make('CartPole-v0')
	num_actions = env.action_space.n

	sample_obs= tf.constant( [env.reset()], tf.float32)
	obs_shape = sample_obs.shape

	network = get_network(obs_shape, num_actions, 24)

	episode_actions, episode_rewards = play_one(env, network)
	print(episode_actions, episode_rewards)

	returns = get_returns(episode_rewards)

	print(returns)