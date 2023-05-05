"""
An implementation of "Touchscreen typing model"
from Jokinen et al., "Touchscreen typing as optimal supervisory control." CHI 2021.

Original code: https://github.com/aditya02acharya/TypingAgent
"""

import tqdm
import logging
import numpy as np
from os import path
from pathlib import Path

from collections import deque

from ..abstract.agent import Agent
from ..algorithms.q_table import QLearningTable
from ..vision.vision_agent_environment import VisionAgentEnv


class VisionAgent(Agent):

    def __init__(self, layout_config, agent_params, verbose=False):
        self.logger = logging.getLogger(__name__)

        self.env = VisionAgentEnv(layout_config, agent_params)
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
        Function to start agent training. Vision agent uses Tabular Q-learning.
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

    def type_char(self, char, eye_loc):
        """
        eye movement to a single character.
        :param char: character to type.
        :param eye_loc: current eye position.
        """
        self.env.eye_location = eye_loc
        self.env.target = char
        self.env.set_belief()
        a = int(self.agent.choose_action(self.env.belief_state))
        (mt_enc, mt_exec, mt_enc_l), mt, moved = self.env.move_eyes(a)
        coord = self.env.device.convert_to_ij(a)

        return (mt_enc, mt_exec, mt_enc_l), mt, self.env.eye_location, coord, moved

    def type_sentence(self, sentence):
        """
        generate sequence of eye movements for a sentence.
        :param sentence: string for which eye movements have to be made.
        :return: test_data: list with eye and action data for the typed sentence.
        """
        # do the initialisation step.
        self.env.reset()
        test_data = []
        self.logger.debug("Typing: %s" % sentence)

        # append log header.
        test_data.append(["model time", "eyeloc x", "eyeloc y", "action x", "action y", "type"])
        test_data.append([round(self.env.model_time, 4), self.env.eye_location[0], self.env.eye_location[1], "", "",
                          "start"])

        for char in sentence:
            self.env.target = char
            (_, mt_exec, mt_enc_l), mt, _, action, _ = self.type_char(char, self.env.eye_location)

            test_data.append(
                [round(self.env.model_time - mt_enc_l*1000 - mt_exec*1000 + 50, 4), self.env.prev_eye_loc[0],
                 self.env.prev_eye_loc[0], "", "", 'encoding'])
            test_data.append(
                [round(self.env.model_time - mt_enc_l*1000, 4), self.env.eye_location[0], self.env.eye_location[1],
                 action[0], action[1], 'saccade'])

            if mt_enc_l > 0:
                test_data.append([round(self.env.model_time, 4), self.env.eye_location[0], self.env.eye_location[0], "",
                                  "", 'late encoding'])

        return test_data

    def evaluate(self, sentence, **kwargs):
        pass
