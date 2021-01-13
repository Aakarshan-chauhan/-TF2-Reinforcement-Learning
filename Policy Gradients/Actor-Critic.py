import tensorflow as tf
import gym 
from gym import wrappers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import wandb

class Actor(Model):
	def __init__(self, num_actions):
		super(Actor, self).__init__()

		self.num_actions = num_actions
		self.fc1 = Dense(1024, activation='relu')
		self.fc2 = Dense(512, activation='relu')
		self.pi = Dense(num_actions, activation='softmax')
	
	def call(self, obs):
		x = self.fc1(obs)
		x = self.fc2(x)
		pi = self.pi(x)
		return pi

class Critic(Model):
	def __init__(self):
		super(Critic, self).__init__()


		self.fc1 = Dense(1024, activation='relu')
		self.fc2 = Dense(512, activation='relu')
		self.v = Dense(1)
	
	def call(self, obs):
		x = self.fc1(obs)
		x = self.fc2(x)
		v = self.v(x)
		return v

class Agent:
	def __init__(self, obs_dims, num_actions, gamma, learning_rate):
		self.actor = Actor(num_actions)
		self.critic = Critic()

		self.num_actions = num_actions

		self.gamma = gamma
		self.alpha = learning_rate
		self.obs_dims = obs_dims
		self.action= None

		self.optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate)
	
	@tf.function
	def get_action(self, obs):
		obs = tf.reshape(obs, (-1, self.obs_dims))
		pi = self.actor(obs)
		
		action_probs = tfp.distributions.Categorical(probs=pi)
		action = action_probs.sample()

		return action[0]

	@tf.function
	def learn(self, obs, obs_, reward, action, done):
		obs = tf.reshape(obs, (-1, self.obs_dims))
		obs_ = tf.reshape(obs_, (-1 , self.obs_dims))
		reward = tf.cast(reward, dtype=tf.float32)
		
		with tf.GradientTape(persistent=True) as tape:
			pi = self.actor(obs)

			V_ = tf.stop_gradient(self.critic(obs_))
			V = self.critic(obs)

			target = reward + self.gamma * V_ * (1-int(done))

			prob = tfp.distributions.Categorical(probs=pi)
			log_prob = prob.log_prob(action)
			advantage = target - V

			actor_loss = - tf.reduce_mean(log_prob*advantage)
			critic_loss = tf.reduce_sum(advantage**2)

		agrads = tape.gradient(actor_loss, self.actor.trainable_variables)
		vgrads = tape.gradient(critic_loss, self.critic.trainable_variables)


		self.optimizer.apply_gradients(zip(agrads, self.actor.trainable_variables))
		self.optimizer.apply_gradients(zip(vgrads, self.critic.trainable_variables))
		del tape
		

if __name__=="__main__":

	env = gym.make("CartPole-v0")
	OBS_DIMS = env.reset().shape[0]
	N_ACTIONS = env.action_space.n
	GAMMA = 0.99
	ALPHA = 0.0002

	agent = Agent(OBS_DIMS, N_ACTIONS, GAMMA, ALPHA)

	wandb.init(project="Actor Critic CartPole")
	wandb.config.OBS_DIMS = OBS_DIMS
	wandb.config.N_ACTIONS = N_ACTIONS
	wandb.config.GAMMA = GAMMA
	wandb.config.ALPHA = ALPHA

	n_games = 500
	scores = []
	avg_scores = []

	with tqdm.trange(n_games) as t:
		for i in t:
			done = False
			score = 0
			s = env.reset()
			while not done:
				a = agent.get_action(s).numpy()
				s_, r, done, info = env.step(a)
				agent.learn(s, s_, r, a, done)
				
				s = s_

				score += r
			scores.append(score)
			avg_score = np.mean(scores[max(0, i-100): (i+1)])
			avg_scores.append(avg_score)

			wandb.log({"Average Reward":avg_score})
			t.set_description(f"Episode= {i}")
			t.set_postfix(Average_reward = avg_score
				 )
			
			
				
				
			


	plt.title("Actor Critic")
	plt.xlabel("Episodes")
	plt.ylabel("Average Rewards")
	plt.plot(avg_scores, label="Average Reward over 100 episodes")
	plt.legend()
	plt.show()


