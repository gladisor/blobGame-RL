from agents import *
from blobGame import *
from utils import *
import numpy as np

class Dyna_QPlus(Q_Agent):
	def __init__(self, 
		alpha, gamma, epsilon, 
		eps_decay, n_planning, kappa):

		Q_Agent.__init__(self, alpha, gamma, epsilon, eps_decay)
		self.n_planning = n_planning
		self.model = {}

		self.kappa = kappa
		## Time since last state action pair visit
		# (y coord, x coord, action num)
		self.tau = np.zeros((10, 10, 4))

	def update_tau(self, state, action):
		self.tau += 1
		self.tau[state][action] = 0

	def update_model(self, state, action, reward, next_state):
		try:
			self.model[state][action] = (reward, next_state)
		except:
 			self.model[state] = {}
 			self.model[state][action] = (reward, next_state)

	def plan(self):
		for _ in range(self.n_planning):
			states = list(self.model.keys())
			idx = np.random.choice(len(states))
			state = states[idx]

			actions = list(self.model[state])
			action = np.random.choice(actions)

			reward, next_state = self.model[state][action]

			reward += self.kappa*np.sqrt(self.tau[state][action])

			if next_state != -1:
				self.q_update(state, action, reward, next_state)
			else:
				self.terminal_update(state, action, reward)


	def agent_start(self, state):
		self.action = self.choose_action_egreedy(state)
		self.state = state
		return self.action

	def agent_step(self, reward, next_state):
		self.update_tau(self.state, self.action)
		self.q_update(self.state, self.action, reward, next_state)
		self.update_model(self.state, self.action, reward, next_state)
		self.plan()

		self.action = self.choose_action_egreedy(next_state)
		self.state = next_state
		self.epsilon = self.epsilon*self.eps_dec
		return self.action
		
	def agent_end(self, reward):
		self.update_tau(self.state, self.action)
		self.terminal_update(self.state, self.action, reward)
		self.update_model(self.state, self.action, reward, -1)
		self.plan()

if __name__ == "__main__":
	agent = Dyna_QPlus(
		alpha=0.1,
		gamma=0.95,
		epsilon=1,
		eps_decay=0.9998,
		n_planning=5,
		kappa=0.1)

	history = {
	'reward_sum':[],
	'num_steps':[]}

	for episode in range(400):
		env = blobGame()

		state = env.state
		action = agent.agent_start(state)

		reward_sum = 0
		num_steps = 0
		done = False
		while not done:
			reward, next_state, done = env.step(action)
			reward_sum += reward
			num_steps += 1

			if done:
				break

			next_action = agent.agent_step(reward, next_state)
			action = next_action

		agent.agent_end(reward)

		print(f"reward_sum: {reward_sum} epsilon: {agent.epsilon}")
		history['reward_sum'].append(reward_sum)
		history['num_steps'].append(num_steps)

	plotPI(agent)

	plt.plot(history['reward_sum'])
	plt.title('Reward per episode')
	plt.xlabel('Episode')
	plt.ylabel('Reward')
	plt.show()