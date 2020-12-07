import tensorflow as tf
import gym 
from gym import wrappers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import numpy as np
import tqdm
class ActorCritic(Model):
	def __init__(self, num_actions):
		super(ActorCritic, self).__init__()

		self.num_actions = num_actions
		self.fc1 = Dense(256, activation='relu')
		self.fc2 = Dense(256, activation='relu')
		self.v = Dense(1, activation=None)
		self.pi = Dense(num_actions, activation='softmax')
	
	def call(self, obs):
		x = self.fc1(obs)
		x = self.fc2(x)
		v = self.v(x)
		pi = self.pi(x)
		return v, pi

class Agent:
	def __init__(self, obs_dims, num_actions, gamma, learning_rate):
		self.model = ActorCritic(num_actions)
		self.num_actions = num_actions

		self.gamma = gamma
		self.alpha = learning_rate
		self.obs_dims = obs_dims
		self.action= None

		self.optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate)

	def get_action(self, obs):
		obs = tf.reshape(obs, (-1, self.obs_dims))
		v, pi = self.model.predict(obs)

		action_probabilities = tfp.distributions.Categorical(probs = pi)
		action = action_probabilities.sample()
		self.action = action
		return action.numpy()[0]


	def learn(self, obs, obs_, reward, done):
		obs = tf.reshape(obs, (-1, self.obs_dims))
		obs_ = tf.reshape(obs_, (-1 , self.obs_dims))
		reward = tf.convert_to_tensor(reward, dtype=tf.float32)
		
		with tf.GradientTape(persistent=True) as tape:
			V, pi = self.model(obs)
			V_ , _ = self.model(obs_)

			returns = tf.stop_gradient(reward + self.gamma * V_ * (1-int(done)))
			prob = tfp.distributions.Categorical(probs=pi)
			log_prob = prob.log_prob(self.action)
			delta = returns - V

			actor_loss = - log_prob*tf.stop_gradient(delta)
			critic_loss = delta**2

			total_loss = actor_loss + critic_loss
		
		grads1 = tape.gradient(total_loss, self.model.trainable_variables)		
		self.optimizer.apply_gradients(zip(grads1, self.model.trainable_variables))

if __name__=="__main__":
	env = gym.make("CartPole-v1")
	obs_dims = env.reset().shape[0]
	n_actions = env.action_space.n

	agent = Agent(obs_dims=obs_dims, num_actions=n_actions, gamma=0.99, learning_rate=0.0003)

	n_games = 501
	scores = []
	avg_score = []

	with tqdm.trange(n_games) as t:
		for i in t:
			done = False
			score = 0
			s = env.reset()
			while not done:
				a = agent.get_action(s)
				s_, r, done, info = env.step(a)
				agent.learn(s, s_, r, done)
				
				s = s_

				score += r
			scores.append(score)
			avg_score = np.mean(scores[max(0, i-100): (i+1)])
			t.set_description(f"Episode= {i}")
			t.set_postfix(Average_reward = avg_score
				 )
			if i % 50 ==0:
				env2 = wrappers.Monitor(env, 'D:/My C and Python Projects/Repos/Reinforcement-Learning/Policy Gradients/results/ActorCritc/ActorCritic_episode' + str(i) + '/', force=True)

				done = False
				s = env2.reset()
				while not done:
					env2.render()
					a = agent.get_action(s)
					s_, r, done, info = env2.step(a)
					s = s_
				
				env2.close()




