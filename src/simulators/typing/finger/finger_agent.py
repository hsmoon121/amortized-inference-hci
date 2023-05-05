"""
An implementation of "Touchscreen typing model"
from Jokinen et al., "Touchscreen typing as optimal supervisory control." CHI 2021.

Original code: https://github.com/aditya02acharya/TypingAgent
"""

import csv
import tqdm
import logging
import numpy as np
from os import path
from pathlib import Path

from collections import deque

import chainer
import chainerrl
from chainer import serializers
from chainer.backends import cuda

from ..abstract.agent import Agent
from .model import QFunction
from .finger_agent_environment import FingerAgentEnv


class FingerAgent(Agent):

    def __init__(self, layout_config, agent_params, finger, train, verbose=False):
        self.logger = logging.getLogger(__name__)

        self.env = FingerAgentEnv(layout_config, agent_params, finger, train)

        # Agent Configuration.
        optimizer_name = 'Adam' if agent_params is None else agent_params['optimizer_name']
        lr = 0.001 if agent_params is None else agent_params['learning_rate']
        dropout_ratio = 0.2 if agent_params is None else int(agent_params['dropout_ratio'])
        n_units = 512 if agent_params is None else int(agent_params['n_units'])
        device_id = 0 if agent_params is None else int(agent_params['device_id'])
        pre_load = False if agent_params is None else bool(agent_params['pre_load'])
        self.gpu = True if agent_params is None else bool(agent_params['gpu'])
        self.save_path = path.join(Path(__file__).parent.parent, "models/finger")
        gamma = 0.99 if agent_params is None else float(agent_params['discount'])
        replay_size = 10 ** 6 if agent_params is None else int(agent_params['replay_buffer'])
        self.episodes = 1000000 if agent_params is None else int(agent_params['episodes'])
        self.verbose = verbose

        self.q_func = QFunction(embed_size=self.env.observation_space.shape[0],
                                dropout_ratio=dropout_ratio,
                                n_actions=self.env.action_space.n, n_hidden_channels=n_units)

        if pre_load:
            serializers.load_npz(path.join(self.save_path, 'model.npz'), self.q_func)

        if self.gpu:
            self.q_func.to_gpu(device_id)

        if optimizer_name == 'Adam':
            self.optimizer = chainer.optimizers.Adam(alpha=lr)
        elif optimizer_name == 'RMSprop':
            self.optimizer = chainer.optimizers.RMSprop(lr=lr)
        else:
            self.optimizer = chainer.optimizers.MomentumSGD(lr=lr)

        self.optimizer.setup(self.q_func)

        # Use epsilon-greedy for exploration/exploitation with linear decay.
        explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=0.1, end_epsilon=0.01,
                                                                decay_steps=int(self.episodes / 4),
                                                                random_action_func=self.env.action_space.sample)

        # DQN uses Experience Replay.
        replay_buffer = chainerrl.replay_buffers.prioritized.PrioritizedReplayBuffer(capacity=replay_size)

        phi = lambda x: x.astype(np.float32, copy=False)

        # Now create an agent that will interact with the environment.
        self.agent = chainerrl.agents.DoubleDQN(q_function=self.q_func, optimizer=self.optimizer,
                                                replay_buffer=replay_buffer, gamma=gamma, explorer=explorer,
                                                replay_start_size=50000, update_interval=1000,
                                                target_update_interval=1000,
                                                target_update_method='soft', phi=phi)

        self.error_list = deque([0], maxlen=1000)
        self.reward_list = deque([0], maxlen=1000)

        if train:
            chainer.config.train = True
            if self.verbose:
                self.pbar = tqdm.tqdm(total=self.episodes, ascii=True,
                                 bar_format='{l_bar}{n}, {remaining}\n')
            else:
                self.pbar = tqdm.tqdm(total=self.episodes)
        else:
            chainer.config.train = False

    def train(self, episodes):
        """
        Function to start agent training. Finger agent uses Double DQN with Prioritised Experience Reply.
        :param episodes: number of training trials to run.
        """
        progress_bar = ProgressBar(self.pbar, episodes)

        chainerrl.experiments.train_agent_with_evaluation(
            agent=self.agent,
            env=self.env,
            steps=episodes,  # Train the agent for n steps
            eval_n_steps=None,  # We evaluate for episodes, not time
            eval_n_episodes=10,  # 10 episodes are sampled for each evaluation
            eval_interval=1000,  # Evaluate the agent after every 1000 steps
            outdir=self.save_path,  # Save everything to 'data/models' directory
            train_max_episode_len=5,  # Maximum length of each episode
            successful_score=4.5,  # Stopping rule
            logger=self.logger,
            step_hooks=[progress_bar]
        )

    def type_char(self, char, sigma, is_eye_present=False):
        """
        finger movement to a single character.
        :param char: character to type.
        :param sigma: target sigma of finger movement given by supervisor
        :param is_eye_present: true if eyes are on target else false
        """

        # set target char key to press.
        self.env.target = char
        self.env.hit = 0

        # use desired sat of 0.1 (fixed) for pre-trained DQN (from original repo)
        self.env.sat_desired = self.env.sat_desired_list.index(0.1)

        # set vision status.
        self.env.vision_status = is_eye_present

        # calculate finger location estimate.
        self.env.calc_max_finger_loc()

        # set belief with current evidence.
        self.env.set_belief()

        # choose the best action with max Q-values
        action, q_val = self.choose_best_action()

        # act on the action.
        _, reward, done, info = self.env.step(action, sigma)

        return info['mt'], reward, done, q_val

    def type_sentence(self, sentence, sigma, is_eye_present=False):
        """
        Types the sentence using only the finger.
        :param is_eye_present: true if eyes are on target else false
        :param sentence: string to be typed with finger.
        :return: test_data : list with eye and action data for the typed sentence.
        """

        self.env.reset()
        test_data = []
        self.logger.debug("Typing: %s" % sentence)

        # Initial data to append to the test data which will be saved and used in visualization
        test_data.append(["model time", "fingerloc x", "fingerloc y", "action x", "action y", "type"])
        test_data.append([round(self.env.model_time, 4), self.env.finger_location[0], self.env.finger_location[1],
                          "", "", "start"])

        for char in sentence:
            self.env.action_type = 0  # setting ballistic action

            # Keep taking actions with same target until peck or max step reached.
            for i in range(10):
                mt, reward, done, q_val = self.type_char(char, sigma, is_eye_present)

                if self.env.action_type == 0:
                    test_data.append([round(self.env.model_time, 4), self.env.finger_location[0],
                                      self.env.finger_location[1], "", "", 'move'])
                else:
                    test_data.append([round(self.env.model_time, 4), self.env.finger_location[0],
                                      self.env.finger_location[1], self.env.finger_location[0],
                                      self.env.finger_location[1], 'peck'])
                if done:
                    break

        return test_data

    def choose_best_action(self):
        """
        Function to choose best action given action-value map.
        """

        state = self.env.preprocess_belief()
        if self.gpu:
            q_values = cuda.to_cpu(chainer.as_array(
                self.q_func(cuda.to_gpu(state.reshape((1, self.env.observation_space.shape[0])), device=0))
                    .q_values).reshape((-1,)))
        else:
            q_values = chainer.as_array(self.q_func(state.reshape((1, self.env.observation_space.shape[0])))
                                        .q_values).reshape((-1,))

        best_action = np.where(q_values == np.amax(q_values))[0]

        return np.random.choice(best_action), np.amax(q_values)
    
    def evaluate(self, sentence, **kwargs):
        pass

    def load(self):
        """
        Function to load pre-trained data.
        """
        serializers.load_npz(path.join(self.save_path, 'model.npz'), self.q_func)
        chainer.config.train = False


class ProgressBar(chainerrl.experiments.hooks.StepHook):
    """
    Hook class to update progress bar.
    """

    def __init__(self, pbar, max_length):
        self.pbar = pbar
        self.max = max_length

    def __call__(self, env, agent, step):
        self.pbar.update()
        if self.max <= step:
            self.pbar.close()
