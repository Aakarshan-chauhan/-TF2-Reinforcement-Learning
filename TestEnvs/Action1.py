import numpy as np

# action space
class action_space():
	# number of actions
	n = 10
	shape = (1,)
	dtype = np.int64

class obervation_space():
	shape=(1,)
	high = [10.]
	low = [0.]
	dtype = np.float32

	
class Env():
	def __init__(self):
		self.action_space = action_space()
		self.observation_space = obervation_space()
		self.timestep = 0

	def reset(self):
		obs = np.random.randint(low=self.observation_space.low[0], high=self.observation_space.high[0], dtype=np.float32)
		self.done = False
		self.timestep = 0
		return np.array(obs)

	def step(self, action):
		self.timestep += 1

		obs = np.random.randint(low=self.observation_space.low[0], high=self.observation_space.high[0], dtype=np.float32)
		if action ==1 :
			reward = 1
		else:
			reward = -1
		
		if self.timestep >= 10:
			self.done=True

		info = str(self.timestep)
		return np.array(obs), reward, self.done, info
