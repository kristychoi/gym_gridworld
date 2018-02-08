"""
Gym implementation of small grid world, where states are represented
as one-hot encodings
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding

class OneHotGridWorldEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		pass

	def _seed(self):
		pass

	def _step(self):
		pass

	def _take_action(self):
		pass

	def _get_reward(self):
		pass

	def in_bounds(self, state):
		pass

	def get_state(self):
		pass

	def get_status(self):
		pass

	def _reset(self):
		pass

	def _render(self, mode='human', close=False):
		raise NotImplementedError