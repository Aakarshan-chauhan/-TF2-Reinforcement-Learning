import numpy as np
from Action1 import Env as action1
class gym:
	def __init__(self):
		self.action1 = action1

	def make(self, name):
		if name == 'action1':
			return self.action1

