from agents import *
from blobGame import *
from utils import *
import matplotlib.pyplot as plt

agent = Dyna_Q(
	alpha=0.1,
	gamma=0.95,
	epsilon=1,
	eps_dec=.9999,
	n_planning=0)

history = {
	'reward_sum':[],
	'num_steps':[]}
	
for episode in range(500):
	env = blobGame()

	state = env.state
	action = agent.agent_start(state)

	reward_sum = 0
	num_steps = 0
	terminal = False

	while not terminal:
		reward, next_state, terminal = env.step(action)
		reward_sum += reward
		num_steps += 1

		agent.update_model(state, action ,reward, next_state, terminal)

		if terminal:
			break

		next_action = agent.agent_step(reward, next_state)
		agent.plan()

		state = next_state
		action = next_action

	print(f"reward_sum: {reward_sum} epsilon: {agent.epsilon}")

	history['reward_sum'].append(reward_sum)
	history['num_steps'].append(num_steps)

	## Terminal update
	agent.agent_end(reward)
	## Final planning step
	agent.plan()

plotPI(agent)

plt.plot(history['reward_sum'])
plt.title('Reward per episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()