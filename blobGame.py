import numpy as np
import cv2
import time

SIZE = 10

## Reward/Penalty
MOVE_PENALTY = -1
ENEMY_PENALTY = -300
FOOD_REWARD = 50
EXIT_REWARD = 300

## Info for assigning colors to the different objects
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
EXIT_N = 4

class Blob():
	def __init__(self, x, y):
		self.x = x
		self.y = y

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
		elif self.x > SIZE-1:
			self.x = SIZE-1

		if self.y < 0:
			self.y = 0
		elif self.y > SIZE-1:
			self.y = SIZE-1

class blobGame():
	def __init__(self):
		self.player = Blob(0, 0)
		self.foods = [(4, 3), (2, 9), (8, 7), (2,6)]
		self.enemys = [(8, 2), (7, 4), (6, 4), (5, 1)]
		self.exit = (SIZE-1, SIZE-1)
		self.state = (self.player.x, self.player.y)

	def eating(self):
		if self.state in self.foods:
			idx = self.foods.index(self.state)
			self.foods.pop(idx)
			return True

	def dead(self):
		if self.state in self.enemys:
			return True
		return False

	def onExit(self):
		if self.state == self.exit:
			return True
		return False

	def step(self, action):
		state = self.state
		self.player.action(action)
		self.state = (self.player.x, self.player.y)
		done = False
		if self.eating():
			reward = FOOD_REWARD
		elif self.dead():
			reward = ENEMY_PENALTY
			done = True
		elif self.onExit():
			reward = EXIT_REWARD
			done = True
		else:
			reward = MOVE_PENALTY
		return state, reward, done