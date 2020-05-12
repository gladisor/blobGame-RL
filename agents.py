import numpy as np

class Q_Agent():
	def __init__(self, 
			alpha, gamma, epsilon, eps_dec):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_dec = eps_dec

		self.grid_size = 10
		self.n_actions = 4

		self.q = np.zeros((self.grid_size, self.grid_size, self.n_actions))

		self.state = None
		self.action = None

	def choose_action_egreedy(self, state):
		if np.random.rand() > self.epsilon:
			action = np.argmax(self.q[state])
		else:
			action = np.random.randint(self.n_actions)
		return action

	def q_update(self, state, action, reward, next_state):
		current_q = self.q[state][action]
		target = np.max(self.q[next_state])

		td = reward + self.gamma*target - current_q
		self.q[state][action] += self.alpha*td

	def terminal_update(self, state, action, reward):
		current_q = self.q[state][action]
		td = reward - current_q
		self.q[state][action] += self.alpha*td

	def agent_start(self, state):
		self.action = self.choose_action_egreedy(state)
		self.state = state
		return self.action

	def agent_step(self, reward, next_state):
		self.q_update(self.state, self.action, reward, next_state)
		self.action = self.choose_action_egreedy(next_state)
		self.state = next_state

		self.epsilon = self.epsilon*self.eps_dec
		return self.action

	def agent_end(self, reward):
		## Terminal update differs from q_update
		self.terminal_update(self.state, self.action, reward)

class Dyna_Q(Q_Agent):
	def __init__(self,
		alpha, gamma, epsilon, eps_dec, n_planning):
		Q_Agent.__init__(self,alpha, gamma, epsilon, eps_dec)
		self.n_planning = n_planning
		self.model = {}

	def update_model(self, state, action, reward, next_state, terminal):
		try:
			self.model[state][action] = (reward, next_state, terminal)
		except:
			self.model[state] = {}
			self.model[state][action] = (reward, next_state, terminal)

	def plan(self):
		for _ in range(self.n_planning):
			states = list(self.model.keys())
			idx = np.random.choice(len(states))
			state = states[idx]

			actions = list(self.model[state])
			action = np.random.choice(actions)

			reward, next_state, terminal= self.model[state][action]

			if not terminal:
				self.q_update(state, action, reward, next_state)
			else:
				self.terminal_update(state, action, reward)

if __name__ == "__main__":
	agent = Dyna_Q(
		alpha=0.1,
		gamma=0.95,
		epsilon=1,
		eps_dec=0.9998,
		n_planning=5)

	agent.update_model((1, 4), 2, -300, (9, 8))
	agent.update_model((2, 4), 3, 50, (4, 6))
	agent.update_model((1, 5), 2, 300, (3, 1))
	agent.update_model((6, 2), 0, -1, (6, 4))
	agent.update_model((3, 8), 1, -1, (8, 1))
	agent.update_model((1, 3), 3, -1, (9, 8))
	agent.plan()