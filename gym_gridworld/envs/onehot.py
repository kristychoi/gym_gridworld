"""
Gym implementation of small grid world, where states are represented
as one-hot encodings
"""
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


class OneHotGridWorldEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.start = 0
		self.terminal_d = {5: 1, 15: 10}  # key indexes position of 1

		# action space of 4 discrete actions
		self.action_space = spaces.Discrete(4)

		self.state = self.start
		self.violation = False
		self.done = False
		self.penalty = -0.1
		# set seed
		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _step(self, action):
		"""
		Takes a step in given environment; returns obs, reward, done, info
		"""
		self._take_action(action)
		reward = self._get_reward()
		obs = self.get_state()
		episode_over = self.get_status()

		return obs, reward, episode_over, {}

	def _take_action(self, action):
		"""
		take given action <action>, then update next state if doesn't go past
		boundaries of given grid world
		"""
		assert not self.done
		state = self.state

		if action == 0: # right
			next_state = state + 1
		elif action == 1: # up
			next_state = state - 4
		elif action == 2: # left
			next_state = state - 1
		else: # down
			next_state = state + 4
		# if next_state is out of bounds, incurs a penalty and stays in current state
		if 0 <= next_state <= 15:
			self.state = next_state
		else:
			self.violation = True

	def _get_reward(self):
		"""
		returns reward for resulting state after taking an action
		"""
		if self.state in self.terminal_d:
			self.done = True
			return self.terminal_d[self.state]
		elif self.violation:
			self.violation = False
			return self.penalty
		else:
			return 0

	def get_state(self):
		"""
		takes the position of the 1 (self.state) and turns this
		into a one-hot encoding representation of the current state
		"""
		state_vec = np.zeros(16)
		state_vec[self.state] = 1
		return state_vec

	def get_status(self):
		return self.done

	def _reset(self):
		self.state = self.start
		self.done = False

	def _render(self, mode='human', close=False):
		raise NotImplementedError