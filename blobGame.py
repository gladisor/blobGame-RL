import numpy as np

class Blob():
	def __init__(self, x, y, grid_size=10):
		self.x = x
		self.y = y
		self.grid_size = grid_size

	def __str__(self):
		return f"{self.x},{self.y}"

	def __sub__(self, other):
		return self.x-other.x, self.y-other.y

	def action(self, direction):
		## Left
		if direction == 0:
			self.move(-1, 0)
		## Up
		elif direction == 1:
			self.move(0, -1)
		## Right
		elif direction == 2:
			self.move(1, 0)
		## Down
		elif direction == 3:
			self.move(0, 1)

	def move(self, x, y):
		self.x += x
		self.y += y

		if self.x < 0:
			self.x = 0
		elif self.x > self.grid_size-1:
			self.x = self.grid_size-1

		if self.y < 0:
			self.y = 0
		elif self.y > self.grid_size-1:
			self.y = self.grid_size-1

class blobGame():
	def __init__(self, grid_size=10):
		## Spawn top left
		self.grid_size = grid_size
		self.n_actions = 4
		self.player = Blob(0, 0, grid_size)
		self.foods = [(4, 3), (2, 9), (8, 7), (2, 6), (2, 2), (9, 0)]
		self.enemys = [(8, 2), (7, 4), (6, 4), (5, 1), (0, 6), (2, 8)]
		## Goal bottem right
		self.exit = (grid_size-1, grid_size-1)
		self.state = (self.player.y, self.player.x)

	def eating(self):
		if self.state in self.foods:
			idx = self.foods.index(self.state)
			self.foods.pop(idx)
			return True
		return False

	def dead(self):
		if self.state in self.enemys:
			return True
		return False

	def onExit(self):
		if self.state == self.exit:
			return True
		return False

	def step(self, action):
		self.player.action(action)
		self.state = (self.player.y, self.player.x)
		terminal = False
		
		MOVE_PENALTY = -1
		ENEMY_PENALTY = -300
		FOOD_REWARD = 50
		EXIT_REWARD = 300
		if self.eating():
			reward = FOOD_REWARD
		elif self.dead():
			reward = ENEMY_PENALTY
			terminal = True
		elif self.onExit():
			reward = EXIT_REWARD
			terminal = True
		else:
			reward = MOVE_PENALTY
		return reward, self.state, terminal