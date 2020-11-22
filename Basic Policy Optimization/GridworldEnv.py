import numpy as np
import random

# a discrete state space and action space environment for tabular method benchmarking

class grid:
	def __init__(self, num_blocks, num_goals, num_deads, height, width):
		self.grid = np.zeros((height, width))-1

		self.height = height
		self.width = width
		self.blocks = []
		self.deads = []
		self.goals = []

		self.start = [0, 0]

		self.done = False
		self.inds = [(x, y) for x in range(self.height) for y in range(self.width)]

		for i in range(num_blocks):
			coord = random.sample(self.inds, 1)
			self.inds.remove(coord[0])
			x, y = coord[0]
			self.blocks.append((x, y))
			self.grid[x,y] = 0

		for i in range(num_goals):
			coord = random.sample(self.inds, 1)
			self.inds.remove(coord[0])
			x, y = coord[0]
			self.goals.append((x, y))
			self.grid[x,y] = 10


		for i in range(num_deads):
			coord = random.sample(self.inds, 1)
			self.inds.remove(coord[0])
			x, y = coord[0]
			self.deads.append((x, y))
			self.grid[x,y] = -10



	
	def step(self, action):
		# Up down left right = 0 1 2 3
		old_pos = self.start
		if action == 0:
			self.start[0] += (-1 if self.start[0] > 0 else 0)

		elif action == 1:
			self.start[0] += (1 if self.start[0] < self.height-1 else 0)

		elif action == 2:
			self.start[1] += (-1 if self.start[1] > 0 else 0)

		elif action == 3:
			self.start[1] += (1 if self.start[1] < self.width-1 else 0)
		
		reward = self.grid[self.start[0], self.start[1]]
		if reward == 0:
			self.start = old_pos
			reward = self.grid[self.start[0], self.start[1]]
		
		if reward !=-1:
			done = True

		state = self.start
		return state, reward, self.done, " "

	def reset(self):
		self.start = [0, 0]
		self.done = False

	def print_board(self):
		
		for i in range(self.height):
			for j in range(self.width):
				if self.start != [i, j]:
					print(self.grid[i, j],end="\t|\t")
				else:
					print("PP", end="\t|\t")

			print("\n",end="")

		print("\n\n")
if __name__ == '__main__':
	np.random.seed(122)
	g = grid(2, 1, 1, 4,4)
	print(g.deads, g.blocks, g.goals)
	g.print_board()
	'''
	for i in range(10):
		action = np.random.randint(4)
		print(g.step(action), action)
		g.print_board()'''