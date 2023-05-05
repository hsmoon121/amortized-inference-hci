"""
An implementation of "Touchscreen typing model"
from Jokinen et al., "Touchscreen typing as optimal supervisory control." CHI 2021.

Original code: https://github.com/aditya02acharya/TypingAgent
"""

import os
from pathlib import Path
import sys
import gym
import yaml
import math
import random
import logging
import numpy as np
import pandas as pd

import Levenshtein as lev

from ..abstract.environment import AgentEnv
from ..vision.vision_agent import VisionAgent
from ..finger.finger_agent import FingerAgent
from ..proofread.proofread_agent import ProofreadAgent
from ..display.touchscreendevice import TouchScreenDevice


class SupervisorEnvironment(AgentEnv):
    def __init__(
        self, 
        layout_config, 
        agent_params, 
        train,
        variable_params=False,
        fixed_params=None,
        seed=None,
        n_eval_sentence=100,
    ):
        """
        Free params in menu search model
        1) obs_prob:  uniform(min=0.0, max=1.0)
            > probability of noticing an error directly from finger movement
        2) who_alpha: trucnorm(mean=0.6, std=0.3, min=0.4, max=0.9)
            > finger movements' speed-accuracy bias (for WHo model)
        3) who_k:     trucnorm(mean=0.12, std=0.08, min=0.04, max=0.20)
            > overall motor resources for finger movement (for WHo model)
        """
        self.seeding(seed)
        self.logger = logging.getLogger(__name__)
        self.config_file = None
        config_path = os.path.join(Path(__file__).parent.parent, "configs")
        if os.path.exists(os.path.join(config_path, layout_config)):
            with open(os.path.join(config_path, layout_config), 'r') as file:
                self.config_file = yaml.load(file, Loader=yaml.FullLoader)
                self.logger.info("Device Configurations loaded.")
        else:
            self.logger.error("File doesn't exist: Failed to load %s file under configs folder." % layout_config)
            sys.exit(0)

        if self.config_file:
            self.device = TouchScreenDevice(self.config_file['layout_file'], self.config_file['config'])
            self.user_distance = self.device.device_params['user_distance']
        else:
            self.device = None

        self.vision_agent = VisionAgent(layout_config, agent_params['vision'])
        self.finger_agent = FingerAgent(layout_config, agent_params['finger'], 0, False)
        self.proofread_agent = ProofreadAgent(layout_config, agent_params['proofread'])
        
        self.variable_params = variable_params
        self.labels =  ["obs_prob", "who_alpha", "who_k"]
        self.min_params = [0.0, 0.4, 0.04]
        self.max_params = [1.0, 0.9, 0.20]
        given_params = [0.7, 0.6, 0.12] if fixed_params is None else fixed_params

        self.free_params = dict(
            obs_prob=0.7,
            who_alpha=0.6,
            who_k=0.12,
        )
        self.set_free_params(given_params)
        self.update_free_params()

        self.sat_desired = agent_params['supervisor']['sat_desired']
        self.found_reward = agent_params['supervisor']['reward']
        self.agent_id = 0
        self.eye_loc = None
        self.prev_eye_loc = None
        self.finger_loc = None
        self.belief = None
        self.finger_q = 0.0
        self.proofread_q = 0.0
        self.is_error = None
        self.sentence_to_type = None
        self.key_found = False
        self.typed = None
        self.eye_on_keyboard = None
        self.is_terminal = False
        self.n_chars = 0
        self.line = ''
        self.typed_detailed = ""
        self.eye_model_time = 0
        self.finger_model_time = 0
        self.eye_on_kb_time = 0

        # Observation
        obs_dim = 2 # Two Q-values from FingerAgent and ProofreadAgent
        feature_dim = len(self.labels) + obs_dim if variable_params else obs_dim
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0,
            shape=(feature_dim,)
        )

        # Action
        action_dim = 3 # Desired sigma / Confidence for typing / Confidence for proofread
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(action_dim,)
        )

        self.mt = 0
        self.model_time = 0
        self.steps = 0
        self.n_fixations_freq = 0
        self.gaze_shift_kb_to_txt_freq = 0
        self.gaze_shift_txt_to_kb_freq = 0
        self.n_back_space_freq = 0
        self.immediate_backspace_freq = 0
        self.delay_backspace_freq = 0
        self.finger_travel_dist = 0
        self.saccade_time = 0
        self.encoding_time = 0
        self.chunk = 0
        self.fixation_duration = []
        self.chunk_length = []
        self.train = train
        self.belief_state = None

        corpus_path = os.path.join(Path(__file__).parent.parent, "corpus", "english")
        self.data = pd.read_csv(os.path.join(corpus_path, "train_corpus.csv"), sep=",")
        self.set_corpus(n_eval_sentence)

        # Load each agent
        self.vision_agent.agent.load()
        self.finger_agent.load()
        self.proofread_agent.agent.load()

    def seeding(self, seed=None):
        self.random_param = np.random.default_rng(seed)

    def set_corpus(self, n_eval_sentence=100, random_sample=False):
        self.eye_test_data = []
        self.finger_test_data = []
        self.eye_viz_log = []
        self.finger_viz_log = []
        self.typing_viz_log = []
        self.sentence_test_data = []

        if not self.train:
            self.eye_test_data.append(["model time", "eyeloc x", "eyeloc y", "action x", "action y", "type"])
            self.finger_test_data.append(["model time", "fingerloc x", "fingerloc y", "action x", "action y", "type"])
            self.eye_viz_log.append(['subject_id', 'sentence_n', 'trialtime', 'x', 'y'])
            self.finger_viz_log.append(['subject_id', 'block', 'sentence_n', 'trialtime', 'x', 'y'])
            self.typing_viz_log.append(['subject_id', 'sentence_n', 'trialtime', 'event'])
            self.sentence_test_data.append([
                "sentence.id", "agent.id", "target.sentence", "wpm", "lev.distance",
                "gaze.shift", "bs", "immediate.bs", "delayed.bs",
                "gaze.keyboard.ratio", "fix.count", "finger.travel", "iki", "correct.error",
                "uncorrected.error", "fix.duration", "chunck.length",
                "error.rate", "kspc", "sentence.length"
            ])
            if random_sample:
                sampled_idx = np.random.choice(self.data.shape[0], n_eval_sentence, replace=False)
                self.sentences_id = self.data.iloc[sampled_idx, 0].values
                self.sentences = list(self.data.iloc[sampled_idx, 1].values)
            else:
                # assuming the 1st column is sentence id
                self.sentences_id = self.data.iloc[:n_eval_sentence, 0].values
                self.sentences = list(self.data.iloc[:n_eval_sentence, 1].values)
            self.sentences_bkp = self.sentences.copy()
        else:
            self.sentences = list(self.data.iloc[n_eval_sentence:, 1].values)

    def update_model_time(self, delta):
        """
        Function to update model runtime.
        :param delta: time to increment in ms.
        """
        self.model_time += delta

    def make_eye_movement(self, char):
        """
        Function to perform eye movement from current position to target char.
        :param char: target character to move eyes to.
        :return movement time in seconds.
        """

        self.prev_eye_loc = self.eye_loc
        # movement values are in seconds. But, for visualisation we use ms. Converting everything here.
        (mt_enc, mt_exec, mt_enc_l), self.mt, self.eye_loc, _, moved = self.vision_agent.type_char(char, self.eye_loc)

        if moved:
            self.n_fixations_freq += 1

        self.fixation_duration.append(self.eye_model_time - self.saccade_time)
        self.eye_model_time += (self.mt * 1000)
        self.saccade_time = round(self.eye_model_time - (mt_enc_l * 1000), 4)

        if not self.train:
            self.eye_test_data.append(
                [round(self.eye_model_time - (mt_enc_l * 1000) - (mt_exec * 1000) + 50, 4), self.prev_eye_loc[0],
                 self.prev_eye_loc[1], "", "", 'encoding'])
            self.eye_test_data.append(
                [round(self.eye_model_time - (mt_enc_l * 1000), 4), self.prev_eye_loc[0], self.prev_eye_loc[1],
                 self.eye_loc[0], self.eye_loc[1], 'saccade'])
            if mt_enc_l > 0:
                self.eye_test_data.append(
                    [round(self.eye_model_time, 4), self.eye_loc[0], self.eye_loc[1], "", "", 'late encoding'])

            self.eye_viz_log.append([self.agent_id,
                                     self.sentences_id[self.sentences_bkp.index(self.sentence_to_type[:-1])],
                                     round(self.eye_model_time, 4), self.eye_loc[0], self.eye_loc[1]])

        return self.mt

    def make_finger_movement(self, char, sigma_desired, eye_time):
        """
         Function to perform finger movement from current position to target char.
        """
        finger_time = 0
        self.mt, _, _, self.finger_q = self.finger_agent.type_char(char, sigma_desired, self.eye_on_keyboard)
        self.finger_loc = self.finger_agent.env.finger_location
        self.finger_travel_dist += self.finger_agent.env.dist
        finger_time += self.mt

        # Keep iterating the finger model until a peck is performed and appending the finger data
        while self.finger_agent.env.action_type == self.finger_agent.env.actions[0]:
            self.finger_model_time += (self.mt * 1000)
            if not self.train:
                self.finger_test_data.append(
                    [round(self.finger_model_time, 4), self.finger_loc[0], self.finger_loc[1], "", "", "move"])
                self.finger_viz_log.append([self.agent_id,
                                            self.sentences_id[self.sentences_bkp.index(self.sentence_to_type[:-1])],
                                            2, round(self.finger_model_time, 4), self.finger_loc[0],
                                            self.finger_loc[1]])

            self.mt, _, _, self.finger_q = self.finger_agent.type_char(char, sigma_desired, self.eye_on_keyboard)
            self.finger_loc = self.finger_agent.env.finger_location
            self.finger_travel_dist += self.finger_agent.env.dist
            finger_time += self.mt

        self.finger_model_time += (self.mt * 1000)
        if not self.train:
            self.finger_test_data.append(
                [round(self.finger_model_time, 4), self.finger_loc[0], self.finger_loc[1], self.finger_loc[0],
                 self.finger_loc[1], "peck"])
            self.finger_viz_log.append([self.agent_id,
                                        self.sentences_id[self.sentences_bkp.index(self.sentence_to_type[:-1])],
                                        2, round(self.finger_model_time, 4), self.finger_loc[0],
                                        self.finger_loc[1]])
            self.typing_viz_log.append([self.agent_id,
                                        self.sentences_id[self.sentences_bkp.index(self.sentence_to_type[:-1])],
                                        round(self.finger_model_time, 4), 'pressed key'])

            # finger waits for eye and eye waits for finger. Next action taken when both have reacted for a target.
            if self.finger_model_time > self.eye_model_time:
                self.eye_on_keyboard = True if not tuple(self.eye_loc) in self.proofread_agent.env.proof_locs else False
                if self.eye_on_keyboard:
                    self.eye_on_kb_time += (self.finger_model_time - self.eye_model_time)
                self.eye_model_time = self.finger_model_time
                self.eye_test_data.append(
                    [round(self.eye_model_time, 4), self.eye_loc[0], self.eye_loc[1], "", "", 'wait'])
            elif self.finger_model_time < self.eye_model_time:
                self.finger_model_time = self.eye_model_time
                self.finger_test_data.append(
                    [round(self.finger_model_time, 4), self.finger_loc[0], self.finger_loc[1], "", "", "wait"])

        # See what is typed and update
        letter = self.device.get_character(self.finger_loc[0], self.finger_loc[1])

        if letter == '<':
            self.typed = self.typed[:-1]
            self.typed_detailed += '<'
        elif letter != '>':
            self.typed += letter
            self.chunk += 1
            self.typed_detailed += letter
        else:
            # pressed enter key. this is terminal state.
            self.typed += letter
            self.typed_detailed += letter
            self.is_terminal = True

    def step(self, action, verbose=False):
        """
        Perform a single step in the environment.
        Args:
            action: 3-dim contiunous space (normazlied)
            action[0]: controlling desired sigma
            action[1]: confidence of typing
            action[2]: confidence of proofread

        Returns:
            state: current finger and proofread agent q values.
            reward: scalar reward value for taking action.
            terminal: if episode is done or not.
            info: dictionary of episode info.
        """
        self.steps += 1
        self.hit_next_char = False
        if self.sentence_to_type.startswith(self.typed):
            # Updates what is left to type
            self.line = self.sentence_to_type.replace(self.typed, '', 1)
        char = self.line[0]

        if len(action.shape) > 1:
            assert action.shape[0] == 1
            action = action[0]
        
        # Sigma examples (corresponding error_chance)
        # 0.1 (0.00006%) / 0.2 (1.24%) / 0.3 (9.55%) / 0.4 (21%) / 0.5 (32%)
        # Supervisor agent controls desired sigma within a range of [0.2, 0.4]
        min_sigma = 0.2
        max_sigma = 0.4
        sigma_desired = action[0] * (max_sigma - min_sigma) + min_sigma
        type_score = action[1] * 8 - 4
        proofread_score = action[2] * 8 - 4

        # Stochastically determine whether to type or proofread
        do_type = ((np.exp(type_score) / (np.exp(type_score) + np.exp(proofread_score))) > np.random.uniform())

        if do_type:
            # Typing action was selected.
            self.logger.debug("Typing character %s, with desired SAT: %.2f" % (char, sigma_desired))

            # EYE MOVEMENT
            eye_time = self.make_eye_movement(char)

            # FINGER MOVEMENT
            self.eye_on_keyboard = True if not tuple(self.eye_loc) in self.proofread_agent.env.proof_locs else False
            if self.eye_on_keyboard:
                self.eye_on_kb_time += (eye_time * 1000)
            self.make_finger_movement(char, sigma_desired, eye_time)

            self.hit_next_char = self.finger_agent.env.hit and not self.is_error
            if not self.is_error:
                self.is_error = not self.finger_agent.env.hit

            # update proofread
            error_chance = 1 + math.erf(-0.5 / sigma_desired / math.sqrt(2))
            self.update_proofread(error_chance)

        else:
            # Eye movement for proofread
            self.logger.debug("Proofreading, with desired SAT: %.2f" % sigma_desired)

            self.chunk_length.append(self.chunk)
            self.chunk = 0

            # Check if eyes are on keyboard
            self.eye_on_keyboard = True if not tuple(self.eye_loc) in self.proofread_agent.env.proof_locs else False

            self.prev_eye_loc = self.eye_loc
            (mt_enc, mt_exec, mt_enc_l), self.mt, self.eye_loc = self.proofread_agent.proofread_text(self.eye_loc)

            if tuple(self.eye_loc) in self.proofread_agent.env.proof_locs and self.eye_on_keyboard:
                self.gaze_shift_kb_to_txt_freq += 1
            self.n_fixations_freq += 1

            self.eye_model_time += (self.mt * 1000)
            if not self.train and self.finger_model_time < self.eye_model_time:
                self.finger_model_time = self.eye_model_time
                self.finger_test_data.append(
                    [round(self.finger_model_time, 4), self.finger_loc[0], self.finger_loc[1], "", "", "wait"])

            if not self.train:
                self.eye_test_data.append(
                    [round(self.eye_model_time - mt_enc_l - mt_exec + 50, 4), self.prev_eye_loc[0],
                     self.prev_eye_loc[1], "", "", 'encoding'])
                self.eye_test_data.append(
                    [round(self.eye_model_time - mt_enc_l, 4), self.prev_eye_loc[0], self.prev_eye_loc[1],
                     self.eye_loc[0], self.eye_loc[1], 'saccade'])
                if mt_enc_l > 0:
                    self.eye_test_data.append(
                        [round(self.eye_model_time, 4), self.eye_loc[0], self.eye_loc[1], "", "",
                         'late encoding'])
                self.eye_viz_log.append([self.agent_id,
                                         self.sentences_id[self.sentences_bkp.index(self.sentence_to_type[:-1])],
                                         round(self.eye_model_time, 4), self.eye_loc[0], self.eye_loc[1]])

            # Once proofread, reset error prob. to lowest value
            self.proofread_agent.env.reset_error_prob()
            self.proofread_agent.env.set_belief()
            self.proofread_q = self.proofread_agent.get_q_value()

            # If errors are found during proofreading
            if self.is_error:
                if len(self.typed) > len(self.sentence_to_type):
                    indexes = [i for i in range(len(self.sentence_to_type)) if
                               self.typed[i] != self.sentence_to_type[i]]
                else:
                    indexes = [i for i in range(len(self.typed)) if self.typed[i] != self.sentence_to_type[i]]
                if len(indexes) > 0:
                    err_idx = indexes[0]
                else:
                    err_idx = len(self.typed)

                # Calculating required back spaces to remove errors
                n_bs = len(self.typed) - err_idx

                self.n_back_space_freq += n_bs

                if n_bs == 1:
                    self.immediate_backspace_freq += 1
                if n_bs > 1:
                    self.delay_backspace_freq += 1

                # For number of required backspaces
                is_error_bs = False
                for bs in range(n_bs):
                    char = '<'
                    # EYE MOVEMENT
                    eye_time = self.make_eye_movement(char)
                    if self.eye_on_keyboard:
                        self.eye_on_kb_time += (eye_time * 1000)

                    # FINGER MOVEMENT
                    self.eye_on_keyboard = True if not tuple(
                        self.eye_loc) in self.proofread_agent.env.proof_locs else False
                    self.make_finger_movement(char, sigma_desired, eye_time)

                    if not self.finger_agent.env.hit:
                        is_error_bs = True

                # Errors corrected after proofreading
                self.is_error = is_error_bs

        total_mt = (self.eye_model_time + self.finger_model_time) / 1000.0
        reward = self.reward(action, total_mt)

        if verbose:
            print(self.typed, self.belief_state[-1], action, do_type, reward)

        self.set_belief()

        if self.is_terminal and not self.train:
            
            if verbose:
                print("-"*10)
                print(self.typed)
                print(self.sentence_to_type)
                print(self.steps, reward, self.n_chars)
                print("WPM", ((len(self.typed)) / 5.0) / (self.finger_model_time / 60000.0))
                print("-"*10)

            # log sentence level data.
            self.logger.debug("typed: %s" % self.typed)
            index = self.sentences_bkp.index(self.sentence_to_type[:-1])
            corrected = 0
            uncorrected = 0
            if self.typed == self.sentence_to_type and self.n_back_space_freq > 0:
                corrected = 1

            if not (self.typed == self.sentence_to_type):
                uncorrected = 1

            if len(self.typed) == 0:
                wpm = 0
                iki = 0
            else:
                wpm = ((len(self.typed)) / 5.0) / (self.finger_model_time / 60000.0)
                iki = self.finger_model_time / len(self.typed_detailed)

            err_lev_dist = lev.distance(self.sentence_to_type, self.typed)
            proportion_gaze_kb = self.eye_on_kb_time / self.eye_model_time
            f_dur = np.mean(self.fixation_duration) if len(self.fixation_duration) > 0 else 0.0
            c_ln = np.mean(self.chunk_length) if len(self.chunk_length) > 0 else 0.0
            sentence_len = len(self.sentence_to_type) - 1
            typed_len = len(self.typed) - 1
            error_rate = err_lev_dist / max(sentence_len, typed_len) * 100.
            kspc = (len(self.typed_detailed) - 1) / sentence_len
            line = [self.sentences_id[index], self.agent_id, self.sentence_to_type, wpm, err_lev_dist,
                    self.gaze_shift_kb_to_txt_freq, self.n_back_space_freq, self.immediate_backspace_freq,
                    self.delay_backspace_freq, proportion_gaze_kb, self.n_fixations_freq,
                    self.finger_travel_dist, iki, corrected, uncorrected, f_dur, c_ln,
                    error_rate, kspc, sentence_len]
            self.sentence_test_data.append(line)

        if self.variable_params:
            return np.concatenate((self.belief_state, self.z)), reward, self.is_terminal, {}
        else:
            return self.belief_state, reward, self.is_terminal, {}

    def reset(self):
        """
        Function to be called on start of a trial. It resets the environment
        and sets the initial belief state.
        :return: current belief state.
        """
        self.logger.debug("Resetting Environment for start of new trial.")
        self.steps = 0
        self.eye_loc = self.device.start()
        self.logger.debug("Eye initialised to location: {%d, %d}" % (self.eye_loc[0], self.eye_loc[1]))
        self.finger_loc = self.device.start()
        self.logger.debug("Finger initialised to location: {%d, %d}" % (self.finger_loc[0], self.finger_loc[1]))
        self.vision_agent.env.reset()
        self.proofread_agent.env.reset()
        self.finger_agent.env.reset()
        self.finger_agent.env.finger_location = self.finger_loc

        self.vision_agent.env.eye_location = self.eye_loc
        self.proofread_agent.env.eye_location = self.eye_loc
        self.mt = 0
        self.key_found = False
        self.is_terminal = False
        self.eye_on_keyboard = True
        self.typed = ""
        self.typed_detailed = ""
        self.finger_q = 0.0
        self.eye_model_time = 0
        self.finger_model_time = 0
        self.eye_on_kb_time = 0
        self.gaze_shift_kb_to_txt_freq = 0
        self.gaze_shift_txt_to_kb_freq = 0
        self.n_back_space_freq = 0
        self.immediate_backspace_freq = 0
        self.delay_backspace_freq = 0
        self.n_fixations_freq = 0
        self.finger_travel_dist = 0
        self.saccade_time = 0
        self.encoding_time = 0
        self.chunk = 0
        self.fixation_duration.clear()
        self.chunk_length.clear()
        self.is_error = False

        self.proofread_agent.env.reset_error_prob()
        self.proofread_agent.env.set_belief()
        self.proofread_q = self.proofread_agent.get_q_value()

        if self.train:
            self.sentence_to_type = random.choice(self.sentences)
        else:
            self.sentence_to_type = self.sentences.pop(0)
            if len(self.sentences) == 0:
                self.sentences += self.sentences_bkp

        self.sentence_to_type += '>'
        self.logger.debug('typing: %s' % self.sentence_to_type)
        self.line = self.sentence_to_type
        self.n_chars = len(self.sentence_to_type)
        if not self.train:
            # Initial data appended for both eye and finger
            self.eye_test_data.append(
                [round(self.eye_model_time, 4), self.eye_loc[0], self.eye_loc[1], "", "", "start"])

            self.finger_test_data.append(
                [round(self.finger_model_time, 4), self.finger_loc[0], self.finger_loc[1], "", "", "start"])

        self.set_belief()

        if self.variable_params:
            return np.concatenate((self.belief_state, self.z))
        else:
            return self.belief_state

    def reward(self, action, movement_time):
        """
        Function for calculating R(a) = total character in sentence - movement time - levenshtein distance.
        :param action: tuple for finger movement action taken by agent.
        :param movement_time: movement time in seconds for taking action.
        :return: reward: float value to denote goodness of action and action type
        """
        accuracy_weight = 0.5
        if self.steps == 100 and not self.is_terminal:
            # reached max length.
            self.is_terminal = True
            r = (-lev.distance(self.sentence_to_type, self.typed)) * accuracy_weight \
                + (-movement_time) * (1 - accuracy_weight)
        elif not self.is_terminal:
            if self.hit_next_char:
                r = 1.0 * accuracy_weight
            else:
                r = 0.0
        else:
            r = (-lev.distance(self.sentence_to_type, self.typed)) * accuracy_weight \
                + (-movement_time) * (1 - accuracy_weight)
        return r

    def render(self, mode='human'):
        pass

    def set_belief(self):
        """
        Function to update belief state.
        """
        if self.sentence_to_type.startswith(self.typed):
            # Updates what is left to type
            self.line = self.sentence_to_type.replace(self.typed, "", 1)

        if len(self.line) > 0:
            char = self.line[0]
        else:
            char = ">"
        self.finger_agent.env.target = char
        self.finger_agent.env.hit = 0
        self.finger_agent.env.sat_desired = self.finger_agent.env.sat_desired_list.index(0.1)

        # calculate finger location estimate.
        self.finger_agent.env.calc_max_finger_loc()

        # set belief with current evidence.
        self.finger_agent.env.set_belief()

        _, self.finger_q = self.finger_agent.choose_best_action()

        self.belief_state = np.asarray([self.finger_q, self.proofread_q]).astype(np.float32)
        self.logger.debug("current belief state is {%s}" % str(self.belief_state))

    def update_proofread(self, error_chance):
        """
        Function to update proof read belief.
        """
        obs_prob = self.proofread_agent.env.observation_probability

        if self.is_error:
            obs_error = obs_prob
        else:
            obs_error = 1 - obs_prob

        self.proofread_agent.env.error_prob = \
            self.proofread_agent.env.update_error_belief(
                obs_error,
                self.proofread_agent.env.error_prob,
                error_chance
            )
        self.logger.debug("Updated error probability to %.2f" % self.proofread_agent.env.error_prob)
        self.proofread_agent.env.set_belief()
        self.proofread_q = self.proofread_agent.get_q_value()

    def sample_free_params(self):
        assert self.variable_params
        self.z = self.random_param.uniform(low=-1., high=1., size=(len(self.labels),))
        for idx, label in enumerate(self.labels):
            self.free_params[label] = (self.z[idx] + 1) * self.max_params[idx] / 2 + self.min_params[idx]

    def set_free_params(self, given_params):
        for l, v in zip(self.labels, given_params):
            self.free_params[l] = v

        z_list = []
        for i, label in enumerate(self.labels):
            z_list.append(
                (self.free_params[label] - self.min_params[i])
                / (self.max_params[i] - self.min_params[i]) * 2 - 1
            )
        self.z = np.array(z_list)

    def update_free_params(self, free_params=None):
        if free_params is None:
            free_params = self.free_params
        self.vision_agent.env.set_free_params(free_params)
        self.finger_agent.env.set_free_params(free_params)
        self.proofread_agent.env.set_free_params(free_params)