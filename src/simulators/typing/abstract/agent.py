"""
An implementation of "Touchscreen typing model"
from Jokinen et al., "Touchscreen typing as optimal supervisory control." CHI 2021.

Original code: https://github.com/aditya02acharya/TypingAgent
"""

import abc
import numpy as np


class Agent(abc.ABC):

    @abc.abstractmethod
    def train(self, episodes):
        pass

    @abc.abstractmethod
    def evaluate(self, sentence, **kwargs):
        pass
