import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt
import tqdm

class DistLayers(tf.keras.layers.Layer):
	def call(self, logits):
		return tf.random.categorical(logits, 1)

class Model(tf.keras.Model):
	def __init__(self, num_actions):
		super().__init__('mlp_policy')
		self.hidden1 = tf.keras.layers.Dense(128,activation='relu')
		self.hidden2 = tf.keras.layers.Dense(128,activation='relu')

		self.policy = tf.keras.layers.Dense(num_actions)
		self.values = tf.keras.layers.Dense(1)
		self.actions = DistLayers()

	def __call__(self, obs):
		obs = tf.convert_to_tensor(obs)
		obs = tf.reshape(obs, (-1, *state_shape))

		x = self.hidden1(obs)
		x = self.policy(x)

		y = self.hidden2(obs)
		y = self.values(y)

		return x, y

	def get_action(self, obs):
		logits, values = self.__call__(obs)

		x = self.actions(logits)
		return tf.squeeze(x, axis=-1), tf.squeeze(values, axis=-1)

def get_samples(buffer_size=128):
	observations = tf.TensorArray(dtype=tf.float32, size=buffer_size, element_shape=state_shape, name="OBSERVATIONS")
	actions = tf.TensorArray(dtype=tf.float32, size=buffer_size, name="ACTIONS")
	rewards = tf.TensorArray(dtype=tf.float32, size=buffer_size, name="REWARDS")


	obs = env.reset()
	done = False
	
	total_rewards = [0]
	for i in tf.range(buffer_size):
		old_obs = obs
		a, vals = model.get_action(obs)
		a = a[0]
		# update the observations and actions
		observations = observations.write(i, tf.cast(obs, tf.float32))
		actions = actions.write(i, tf.cast(a, tf.float32))

		obs, r, done, info = env.step(a.numpy())
		# update the rewards and done flags

		

		total_rewards[-1]+=r
		rewards = rewards.write(i, tf.cast(total_rewards[-1], tf.float32))
		if done:
			obs = env.reset()
			total_rewards.append(0.)

	observations = observations.stack()
	actions = actions.stack()
	rewards = rewards.stack()

	return observations, actions, rewards , total_rewards

@tf.function
def update(observations, rewards):

	old_obs = observations[:-1]
	obs = observations[1:]

	r = rewards[:-1]
	
	with tf.GradientTape(watch_accessed_variables=True) as tape:
		v_next = model.get_action(obs)[1]
		v = model.get_action(old_obs)[1]
		G = tf.stop_gradient(r + gamma * v_next)
		advantage = tf.stop_gradient(G - v)

		logits = model(obs)[0]
		advantage = tf.expand_dims(advantage, 1)

		actor_loss = -tf.reduce_sum(tf.math.log_softmax(logits) *advantage)
		critic_loss = tf.keras.losses.MSE(G, v)

	grads = tape.gradient([actor_loss, critic_loss], model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))


if __name__=="__main__":
	env = gym.make("CartPole-v1")
	num_actions = env.action_space.n
	state_shape = env.reset().shape
	model = Model(num_actions)
	optimizer = tf.keras.optimizers.Adam()
	rews = []


	gamma = 0.99

	with tqdm.trange(5000) as t:
		for i in t:
			observations, actions, rewards, total_rews = get_samples(100)
			update(observations, rewards)
			rews.extend(total_rews)

			t.set_description(f"Episode: {i}")
			t.set_postfix(
				Highest = np.max(rews),
				Current = np.mean(total_rews),
				Average = (np.mean(rews[i:i+100]) if i >=100 else np.mean(rews))
				)

	avg_100_rewards = [np.mean(rews[i:i+100]) for i in range(len(rews)- 100)]
	fig = plt.figure()
	fig.patch.set_facecolor('black')
	plt.xlabel('Updates')
	plt.ylabel('Rewards')

	plt.plot(rews, label="Rewards", color="dimgray")
	plt.plot(avg_100_rewards, label="Running Average", color="violet")
	ax = plt.gca()
	ax.set_facecolor("black")
	
	plt.legend()
	plt.show()