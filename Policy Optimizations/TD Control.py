import numpy as np
import tqdm
import matplotlib.pyplot as plt

num_actions = 10
class testEnv:
	def __init__(self):
		self.state = np.random.randint(0,10)
		self.steps = 0
	def step(self, action):
		self.steps +=1

		done = False
		if self.steps == 10:
			done = True

		if action == self.state:
			self.state= np.random.randint(0,10)
			return self.state, 1, done, None
		else:
			self.state= np.random.randint(0,10)
			return self.state, -1, done, None

	def reset(self):
		self.state = np.random.randint(0,10)
		self.steps = 0
		return self.state

	def sample(self):
		return np.random.randint(num_actions)


def init_Q(state):
	try:
		x = Q[state]
	except:
		Q[state] = np.zeros(num_actions)

def get_action(state):
	if np.random.random() > eps:
		return np.argmax(Q[state],0)
	else:
		return env.sample()

def update_policy(s, a, r, s_n, a_n):
	Q[s][a] = Q[s][a] + alpha * (r + gamma*Q[s_n][a_n] - Q[s][a])
def play_one():
	s = env.reset()
	done = False

	init_Q(s)
	
	a_next = get_action(s)

	rewards = []
	while not done:
		old_s = s
		a = a_next

		s, r, done, info = env.step(a)

		init_Q(s)

		a_next = get_action(s)

		update_policy(old_s, a, r, s, a_next)	
		rewards.append(r)
	return np.sum(rewards)
if __name__ == "__main__":
	env = testEnv()
	Q = {}

	gamma = 0.99
	alpha = 0.01
	eps = 0.01
	rews = []
	for i in tqdm.tqdm(range(8000)):
		rews.append(play_one())
		if i > 5000:
			eps = 0
	avg_100_rewards = [np.mean(rews[i:i+100]) for i in range(len(rews)- 100)]
	plt.xlabel('Episodes')
	plt.ylabel('Rewards')

	plt.plot(avg_100_rewards)

	plt.show()