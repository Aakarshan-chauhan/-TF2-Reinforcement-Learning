import numpy as np
import matplotlib.pyplot as plt
import tqdm
num_actions = 4

def init_Q(state):
	try:
		temp = Q[(state, 1)]
	except:
		for a in acts:
			Q[(state, a)] = 0.

class testEnv:
	def __init__(self):
		self.state = np.random.randint(4)
		self.steps = 0
	def step(self, action):
		self.steps +=1

		done = False
		if self.steps == 10:
			done = True

		if action == self.state:
			self.state= np.random.randint(10)
			return self.state, 1, done, None
		else:
			self.state= np.random.randint(10)
			return self.state, -1, done, None

	def reset(self):
		self.state = np.random.randint(10)
		self.steps = 0
		return self.state

	def sample(self):
		return np.random.randint(num_actions)


def get_action(state):

	if np.random.random() < eps:
		return env.sample()
	for a in acts:
		try:
			maxs, besta = ((Q[(state, a)], a) if Q[(state,a)] > maxs else (maxs, besta))
		except:
			maxs = Q[(state, a)]
			besta = a

	return besta

def get_returns(states, rewards):
	g = {}
	for s in states:
		g[s] = 0.
	gprev = 0
	for s, r in zip(reversed(states), reversed(rewards)):
		g[s] = r + 0.99 * gprev
		gprev = g[s]
	return g

def update_Q(states, actions, rewards, returns):
	for i in reversed(range(len(states))):
		if i == len(states)-1:
			Q[(states[i], actions[i])] = Q[(states[i], actions[i])] + (returns[states[i]])/N[states[i]]
	

		else:
			Q[(states[i], actions[i])] = Q[(states[i], actions[i])] + (returns[states[i]] - Q[(states[i+1], actions[i+1])])/N[states[i]]
	
def play():

	states = []
	actions = []
	rewards = []
	s = env.reset()
	done = False
	init_Q(s)
	states.append(s)
	try:
		N[s] +=1
	except: 
		N[s] = 1

	while not done:
		a = get_action(s)
		s, r, done, info = env.step(a)
		states.append(s)
		rewards.append(r)
		actions.append(a)
		init_Q(s)

		try:
			N[s] +=1
		except: 
			N[s] = 1
	returns = get_returns(states, rewards)
	actions.append(a)
	update_Q(states, actions, rewards, returns)
	return sum(rewards)

if __name__ == "__main__":
	env = testEnv()
	acts = list(range(num_actions))
	Q = {}
	N = {}
	rews = []
	s = env.reset()
	eps = 0.6
	for i in tqdm.tqdm(range(10000)):
		rews.append(play())
		eps = 0.6/6

	avg_100_rewards = [np.mean(rews[i:i+100]) for i in range(len(rews)- 100)]
	plt.xlabel('Episodes')
	plt.ylabel('Rewards')

	plt.plot(avg_100_rewards)

	plt.show()