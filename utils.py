import matplotlib.pyplot as plt
import numpy as np
from blobGame import *

def plotPI(agent):
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

	env = blobGame(10)
	grid_size = env.grid_size

	pi = np.zeros((grid_size, grid_size))
	for y in range(grid_size):
		for x in range(grid_size):
			pi[y, x] = np.argmax(agent.q[y, x])

	image = np.ones((grid_size, grid_size, 3))
	## Plot start
	image[env.player.y][env.player.x] = colors[PLAYER_N]
	## Plot foods
	for food in env.foods:
		image[food] = colors[FOOD_N]
	## Plot enemys
	for enemy in env.enemys:
		image[enemy] = colors[ENEMY_N]
	## Plot exits
	image[env.exit] = colors[EXIT_N]
	## Key for plotting directions
	directions = {0:"←",1:"↑",2:"→",3:"↓"}

	# Adding directions
	plt.imshow(image)
	bot = Blob(0, 0)
	while True:
		if bot.x == 9 and bot.y == 9:
			break
		action = pi[bot.y, bot.x]
		print(action)
		plt.text(bot.x, bot.y, directions[action])
		bot.action(action)

	# plt.imshow(image)
	# for x in range(0, 10):
	# 	for y in range(0, 10):
	# 		plt.text(x, y, directions[pi[y][x]])

	plt.title(f"Agent policy")
	label = "Blue: Start\nGreen: Food\nRed: Enemy\nBlack: Exit"
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	plt.gcf().text(0.02, 0.5, label, fontsize=20, bbox=props)
	plt.show()