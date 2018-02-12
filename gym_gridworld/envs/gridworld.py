"""
Gym implementation of sudarshan's 4x4 grid world
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class GridWorldEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		# specifying size, start_state, terminal_state_d as per OG code
		self.start = (0,0)
		self.terminal_d = {(1, 1):1, (3, 3):10}

		# action space of 4 discrete actions
		self.action_space = spaces.Discrete(4)

		# make sure start state is within grid world boundaries
		assert self.in_bounds(self.start)
		self.terminal_dict = {}
		assert len(self.terminal_d.keys()) > 0

		for state in self.terminal_d:
			assert self.in_bounds(state)

		self.state = self.start
		self.violation = False
		self.done = (self.state in self.terminal_d)
		assert not self.done
		
		self.penalty = -0.1
		# get seed
		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		"""
		take a step in your environment
		"""
		self._take_action(action)
		reward = self._get_reward()
		ob = self.get_state()
		episode_over = self.get_status()
		
		return ob, reward, episode_over, {}

	def _take_action(self, action):
		"""
		taking an action should update the current state of the agent
		"""
		assert not self.done
		state = self.state

		# take specific actions 
		if action == 0: #right
			next_state = (state[0], state[1]+1)
		elif action == 1: #up
			next_state = (state[0]-1, state[1])
		elif action == 2: #left
			next_state = (state[0], state[1]-1)
		else: #down
			next_state = (state[0]+1, state[1])
		# if next state is out of bounds, incurs a penalty. stay in state
		if self.in_bounds(next_state):
			self.state = next_state
		else:
			self.violation = True	
		# doesn't return anything, but sets current state to next state


	def _get_reward(self):
		"""
		this returns the reward that you would get from taking a certain action
		"""
		if self.state in self.terminal_d:
			self.done = True
			return self.terminal_d[self.state]
		elif self.violation:
			self.violation = False  # set this to False now
			return self.penalty
		else:
			return 0  # not sure if you're actually supposed to return 0

	def in_bounds(self, state):
		return (0 <= state[0] < self.action_space.n) and (0 <= state[1] < self.action_space.n)

	def get_state(self):
		return self.state

	def get_status(self):
		return self.done

	def _reset(self):
		self.state = self.start
		self.done = False
		self.get_state()

	def _render(self, mode='human', close=False):
		raise NotImplementedError
