import numpy as np
import matplotlib.pyplot as plt
import tqdm
num_actions = 15

class testEnv:
	def __init__(self):
		self.state = 0
		self.steps = 0
		self.old_action = np.random.randint(5)

	def step(self, action):
		self.steps +=1

		done = False
		if self.steps == 10:
			done = True

		if action >= self.old_action:
			self.state= self.steps
			self.old_action = action
			return self.state, 1, done, None
		else:
			self.state= self.steps
			return self.state, -1, done, None
		
	def reset(self):
		self.steps = 0
		self.state = self.steps
		return self.state

	def sample(self):
		return np.random.randint(num_actions)

def init_Q(state):
	try:
		temp = Q[state]
	except:
		Q[state] = np.zeros(num_actions)


def add_N(state, action):
	try:
		N[state][action] += 1
	except:
		N[state] =np.zeros(num_actions)
		N[state][action] += 1

def get_action(state):

	if np.random.random() < eps:
		return env.sample()
	
	besta = np.argmax(Q[state])

	return besta

def get_returns(states, rewards):
	old_G = 0.0
	G = {}
	for i in reversed(range(len(states))):
		G[states[i]] = rewards[i] + gamma*old_G
		old_G = G[states[i]]
	return G

def update_Q(states, returns, actions):
	for i in range(len(states)):
		step_size = 1./N[states[i]][actions[i]]
		Q[states[i]][actions[i]] = Q[states[i]][actions[i]] + step_size*(returns[states[i]] - Q[states[i]][actions[i]])

def play_one():
	states = []
	actions = []
	rewards = []

	done = False

	s = env.reset()
	init_Q(s)
	states.append(s)

	while not done:
		a = get_action(s)
		actions.append(a)
		add_N(s, a)
		s, r, done, info = env.step(a)
		init_Q(s)
		states.append(s)
		rewards.append(r)
	
	actions.append(a)
	add_N(s,a)
	rewards.append(r)
	G = get_returns(states, rewards)
	update_Q(states, G, actions)
	return np.sum(rewards)

if __name__ == "__main__":
	env = testEnv()
	N = {}
	Q = {}
	gamma = 0.99
	eps = 0.1
	rews= []
	for i in range(20000):
		rews.append(play_one())
		

	avg_10_rewards = [np.mean(rews[i:i+10]) for i in range(len(rews)- 10)]
	plt.xlabel('Episodes')
	plt.ylabel('Rewards')

	plt.plot(avg_10_rewards)

	plt.show()
		