import numpy as np
import tqdm
import matplotlib.pyplot as plt

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


def init_Q(state, action):
	try:
		x = Q[(state, action)]
	except