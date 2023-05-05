"""
An implementation of "Touchscreen typing model"
from Jokinen et al., "Touchscreen typing as optimal supervisory control." CHI 2021.

Original code: https://github.com/aditya02acharya/TypingAgent
"""

import tqdm
import logging
from os import path
from pathlib import Path
from collections import deque

from ..abstract.agent import Agent
from ..algorithms.q_table import QLearningTable
from ..proofread.proofread_agent_environment import ProofreadAgentEnv


class ProofreadAgent(Agent):

    def __init__(self, layout_config, agent_params, verbose=False):
        self.logger = logging.getLogger(__name__)

        self.env = ProofreadAgentEnv(layout_config, agent_params)
        self.discount_factor = agent_params['discount']
        self.learning_rate = agent_params['learning_rate']
        self.epsilon = agent_params['epsilon']
        self.episodes = agent_params['episodes']
        self.agent = QLearningTable(
            actions=self.env.action_space.n,
            learning_rate=self.learning_rate,
            reward_decay=self.discount_factor,
            e_greedy=self.epsilon,
            filepath=path.join(Path(__file__).parent.parent, "models"),
            filename=agent_params['q_table']
        )
        self.error_list = deque([0], maxlen=1000)
        self.reward_list = deque([0], maxlen=1000)
        self.verbose = verbose

    def train(self, episodes):
        """
        Function to start agent training. Proofread agent uses Tabular Q-learning.
        :param episodes: number of training trials to run.
        """
        if self.verbose:
            iter = tqdm.tqdm(iterable=range(episodes), ascii=True,
                             bar_format='{l_bar}{n}, {remaining}\n')
        else:
            iter = tqdm.tqdm(range(episodes))

        for ep in iter:
            s = self.env.reset()
            done = False
            while not done:
                a = int(self.agent.choose_action(s))
                s_, r, d, _ = self.env.step(a)
                self.reward_list.append(r)
                td_error = self.agent.learn(s, a, r, s_, d)
                self.error_list.append(td_error)
                done = d
                s = s_

        self.agent.save()

    def get_q_value(self):
        """
        Function to get the current q-value for the state.
        :return: q_val: scalar q-value.
        """
        return self.agent.get_max_q(self.env.belief_state)

    def proofread_text(self, eye_location):
        """
        Proofread current text typed.
        """
        self.env.eye_location = eye_location
        _, _, _, info = self.env.step(0)

        return info['encoding'], info['mt'], self.env.eye_location

    def evaluate(self, sentence, **kwargs):
        pass
