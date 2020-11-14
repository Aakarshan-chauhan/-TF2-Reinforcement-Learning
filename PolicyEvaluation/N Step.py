import numpy as np
import gym

def init_Q(state):
	try:
		temp = Q[(state, 1)]
	except:
		for a in acts:
			Q[(state, a)] = 0.

def get_action(state):
	eps = 0.1

	if np.random.random() < eps:
		return env.action_space.sample()

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
	
def play():

	states = []
	actions = []
	rewards = []
	s = env.reset()
	done = False
	init_Q(s)
	states.append(s)
	while not done:
		init_Q(s)
		a = get_action(s)
		s, r, done, info = env.step(a)
		states.append(s)
		rewards.append(r)
		actions.append(a)
	returns = get_returns(states, rewards)
	return returns

if __name__ == "__main__":
	env = gym.make('Taxi-v3')
	acts = list(range(env.action_space.n))
	Q = {}

	s = env.reset()
	
	print(play())