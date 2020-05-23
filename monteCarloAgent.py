import numpy as np
import matplotlib.pyplot as plt
from blobGame import *

## Constants
epsilon = 1 ## Percentage of the time we take a random action
EPS_DECAY = 0.9998 ## Rate at which we reduce random actions
alpha = 0.1 ## Learning rate
gamma = 0.95 ## Discount applied to future reward

def generate_episode(pi, epsilon):
	env = blobGame()

	states, actions, rewards = [], [], []
	done = False
	while not done:
		state = env.state
		states.append(state)

		## Select random action epsilon % of the time
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

## Initializing state action pairs
q_table = np.zeros((10, 10, 4))

## Initializing policy
pi = np.zeros((10, 10))

history = []

EPISODES = 20000
## Policy iteration for finding q* and pi*
for episode in range(EPISODES):
	## Play a game according to the current 
	states, actions, rewards = generate_episode(pi, epsilon)
	avg_reward = round(sum(rewards)/len(rewards), 2)
	G = 0
	## Loop backwards through episode
	history.append(avg_reward)
	for i in range(len(states)-1, -1, -1):
		state, action, reward = states[i], int(actions[i]), rewards[i]

		## Expected future reward from current state
		G = reward + gamma*G
		## Old state action value
		Q_St_At = q_table[state[1]][state[0]][action]

		## Bellman Update equation
		q_table[state[1]][state[0]][action] = Q_St_At + alpha*(G - Q_St_At)

	## Decay epsilon
	epsilon = epsilon*EPS_DECAY

	## Set pi to be the greedy action for all states and actions in q_table
	greedify(q_table, pi)
	print(f"Episode: {episode}, Epsilon: {epsilon}, Reward: {avg_reward}")


## Info for assigning colors to the different objects
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
EXIT_N = 4

colors = {
	1:(0/255, 0/255, 204/255),
	2:(153/255, 255/255, 153/255),
	3:(255/255, 153/255, 153/255),
	4:(0, 0, 0)}

def plotPI(pi):
	env = blobGame()
	image = np.ones((10, 10, 3))

	## Plot start
	image[env.player.y][env.player.x] = colors[PLAYER_N]
	## Plot foods
	for food in env.foods:
		image[food[1]][food[0]] = colors[FOOD_N]
	## Plot enemys
	for enemy in env.enemys:
		image[enemy[1]][enemy[0]] = colors[ENEMY_N]
	## Plot exits
	image[env.exit[1]][env.exit[0]] = colors[EXIT_N]

	## Key for plotting directions
	directions = {0:"←",1:"↑",2:"→",3:"↓"}

	## Adding directions
	plt.imshow(image)
	for x in range(0, 10):
		for y in range(0, 10):
			plt.text(x, y, directions[pi[y][x]])

	plt.title(f"Policy after {EPISODES} episodes")
	label = "Blue: Start\nGreen: Food\n Red: Enemy\nBlack: Exit"
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	plt.gcf().text(0.02, 0.5, label, fontsize=20, bbox=props)
	plt.show()

def plotHistory(history):
	plt.plot(history)
	plt.title("Reward per episode")
	plt.xlabel("Episode number")
	plt.ylabel("Reward")
	plt.show()

plotPI(pi)
plotHistory(history)

	 