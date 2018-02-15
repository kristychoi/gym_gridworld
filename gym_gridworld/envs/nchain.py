"""
Gym implementation of N-Chain MDP
"""
import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding


class NChain(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_states, reset_val=0.01):
        self.state = 0
        self.start = 0
        self.done = False

        # define action space
        self.action_space = spaces.Discrete(2)  # possible actions: return to start,
        # advance forward

        # n_states
        self.n_states = n_states
        self.reset_val = reset_val

        # reset
        self._reset()

        # get seed
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        """
        take a step in your environment
        :param action:
        :return:
        """
        self._take_action(action)
        reward = self._get_reward(action)
        obs = self.get_state()
        episode_over = self.get_status()

        return obs, reward, episode_over, {}

    def _take_action(self, action):
        assert not self.done
        state = self.state

        # take specific actions to calculate prospective next state
        if action == 1:  # advance down the chain
            self.state = state + 1
            if self.state == (self.n_states-1):
                self.done = True
        else:  # otherwise go back to start state
            self.state = 0

    def _get_reward(self, action):
        if self.done and action == 1:
            return 1
        elif action == 0:
            return self.reset_val
        else:
            return 1

    def get_state(self):
        """
        return one-hot encoding of current state
        :return:
        """
        state = np.zeros(self.n_states)
        state[self.state] = 1
        return state

    def get_status(self):
        return self.done

    def _reset(self):
        self.state = self.start
        self.done = False
        assert not self.done

        return self.get_state()