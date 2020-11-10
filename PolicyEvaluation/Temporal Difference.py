import numpy as np
import gym

def get_returns(s, r, s_next):
	
	V[s] = V[s] + (r + 0.99*V[s_next] - V[s])/N[s]
	return V

def play(V):
	states = []
	rewards = []
	done = False
	s = env.reset()
	states.append(s)
	try:
		N[s] +=1
	except:
		N[s] = 1
		V[s] = 0
	while not done:

		old_s = s
		s, r, done, info = env.step(env.action_space.sample())

		states.append(s)
		rewards.append(r)

		try:
			N[s] +=1
		except:
			N[s] = 1
			V[s] = 0

		V = get_returns(old_s, r, s)

if __name__ == "__main__":
	env = gym.make("Taxi-v3")
	N = {}
	V = {}
	for i in range(13):

		play(V)
	print(V)
	print(N)