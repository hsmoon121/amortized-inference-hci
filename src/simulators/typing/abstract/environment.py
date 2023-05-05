"""
An implementation of "Touchscreen typing model"
from Jokinen et al., "Touchscreen typing as optimal supervisory control." CHI 2021.

Original code: https://github.com/aditya02acharya/TypingAgent
"""

import gym
import abc


class AgentEnv(gym.Env, abc.ABC):

    def __init__(self):
        self.observation_space = None
        self.action_space = None


    @abc.abstractmethod
    def step(self, action):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def reward(self, action, movement_time):
        pass

    def get_observation_space(self):
        return self.observation_space.shape

    def get_action_space(self):
        return self.action_space.shape

