# Monte Carlo 
import numpy as np
import gym


def get_returns(states, rewards):
	g = {}
	for s in states:
		g[s] = 0.
	gprev = 0.
	for s,r in zip(reversed(states), reversed(rewards)):
		
		g[s] = r + 0.99 * gprev
		
		gprev = g[s]
	return g

def play():
	states = []
	
	rewards = []
	done = False
	s = env.reset()
	states.append(s)
	while not done:
		try:
			N[s] +=1
		except:
			N[s] = 0
		s, r, done, info = env.step(env.action_space.sample())
		states.append(s)
		rewards.append(r)

	returns = get_returns(states,rewards)
	return returns

def get_values(returns):
	for i in returns.keys():
		try:
			V[i] += (returns[i] - V[i])/N[i]
		except:
			V[i] = 0
	return V
if __name__=="__main__":
	env = gym.make('Taxi-v3')
	N = {}
	V = {}
	for i in range(5):
		returns = play()
		V = get_values(returns)
		

	print(V)
	print(N)