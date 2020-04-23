import numpy as np
import matplotlib.pyplot as plt
from blobGame import *

## Constants
epsilon = 0.9
EPS_DECAY = 0.9998
alpha = 0.1
gamma = 0.95

def generate_episode(q_table, pi, epsilon):
	env = blobGame()

	states, actions, rewards = [], [], []
	done = False
	while not done:
		state = env.state
		states.append(state)

		if np.random.rand() > epsilon:
			action = pi[state[1]][state[0]]
		else:
			action = np.random.randint(4)
		actions.append(action)

		state, reward, done = env.step(action)
		rewards.append(reward)
	return states, actions, rewards

def greedify(q_table, pi):
	for x in range(0,10):
		for y in range(0,10):
			pi[y][x] = np.argmax(q_table[y][x])

EPISODES = 20000

q_table = np.zeros((10, 10, 4))
pi = np.zeros((10, 10))

history = []
for episode in range(EPISODES):
	states, actions, rewards = generate_episode(q_table, pi, epsilon)
	epsilon = epsilon*EPS_DECAY
	G = 0
	## Loop backwards through episode
	avg_reward = round(sum(rewards)/len(rewards),2), epsilon
	history.append(avg_reward)
	for i in range(len(states)-1, -1, -1):
		state, action, reward = states[i], int(actions[i]), rewards[i]

		## Expected future reward
		G = reward + gamma*G
		Q_St_At = q_table[state[1]][state[0]][action]
		q_table[state[1]][state[0]][action] = Q_St_At + alpha*(G - Q_St_At)

	greedify(q_table, pi)

	print(epsilon)

ax1=plt.subplot(121)
ax2=plt.subplot(122)

ax1.plot(history)
ax1.set_title
ax2.imshow(pi)

plt.show()