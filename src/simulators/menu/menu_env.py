"""
An implementation of "Menu search model"
from Kangasrääsiö et al., "Inferring cognitive models from data using approximate Bayesian computation." CHI 2017.

Original code: https://github.com/akangasr/cogsciabc
"""

import math
import numpy as np
from enum import IntEnum
import gym
import logging
logger = logging.getLogger(__name__)

from .utils import Path, Transition


class State():
    """ State of MDP observed by the agent

    Parameters
    ----------
    obs_items : list of MenuItems
    focus : Focus
    quit : Quit
    """
    def __init__(self, obs_items, focus, quit):
        self.obs_items = obs_items
        self.focus = focus
        self.quit = quit

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (tuple(self.obs_items) + (self.focus, self.quit)).__hash__()

    def __repr__(self):
        return "({},{},{})".format(self.obs_items, self.focus, self.quit)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return State([item.copy() for item in self.obs_items], self.focus, self.quit)

    def arr(self):
        arr = list()
        for i in range(len(self.obs_items)):
            # Following the CHI'17 paper, state only includes ItemRelevance but ItemLength
            arr.append(int(self.obs_items[i].item_relevance) / 4.0 * 2.0 - 1.0)

        focus_onehot = np.zeros((9,))
        focus_onehot[int(self.focus)] = 1
        arr += list(focus_onehot)
        arr.append(int(self.quit) * 2.0 - 1.0)
        return np.array(arr)


class ItemRelevance(IntEnum):
    NOT_OBSERVED = 0
    TARGET_RELEVANCE = 1  # 1.0
    HIGH_RELEVANCE = 2  # 0.6
    MED_RELEVANCE = 3  # 0.3
    LOW_RELEVANCE = 4  # 0.0


class ItemLength(IntEnum):
    NOT_OBSERVED = 0
    TARGET_LENGTH = 1
    NOT_TARGET_LENGTH = 2


class MenuItem():
    """ Single menu item

    Parameters
    ----------
    item_relevance : ItemRelevance
    item_length : ItemLength
    """
    def __init__(self, item_relevance, item_length):
        self.item_relevance = item_relevance
        self.item_length = item_length

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (int(self.item_relevance), int(self.item_length)).__hash__()

    def __repr__(self):
        return "({},{})".format(self.item_relevance, self.item_length)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return MenuItem(self.item_relevance, self.item_length)


class Focus(IntEnum):  # assume 8 items in menu
    ITEM_1 = 0
    ITEM_2 = 1
    ITEM_3 = 2
    ITEM_4 = 3
    ITEM_5 = 4
    ITEM_6 = 5
    ITEM_7 = 6
    ITEM_8 = 7
    ABOVE_MENU = 8


class Action(IntEnum):  # assume 8 items in menu
    LOOK_1 = 0
    LOOK_2 = 1
    LOOK_3 = 2
    LOOK_4 = 3
    LOOK_5 = 4
    LOOK_6 = 5
    LOOK_7 = 6
    LOOK_8 = 7
    QUIT = 8


class MenuSearchEnv(gym.Env):
    def __init__(
        self,
        variant=3,
        variable_params=False,
        menu_type="semantic",
        menu_groups=2,
        menu_items_per_group=4,
        semantic_levels=3,
        gap_between_items=0.75,
        prop_target_absent=0.1,
        length_observations=False,
        p_obs_len_cur=0.95,
        p_obs_len_adj=0.89,
        max_number_of_actions_per_session=15,
        seed=None,
    ):
        super().__init__()
        self.seeding(seed)
        assert variant in [0, 1, 2, 3]
        self.variant = variant
        self.variable_params = variable_params
        self.menu_type = menu_type
        self.menu_groups = menu_groups
        self.menu_items_per_group = menu_items_per_group
        self.n_items = self.menu_groups * self.menu_items_per_group
        self.max_number_of_actions_per_session = max_number_of_actions_per_session
        assert self.n_items == 8

        self.reward_success = 10000
        self.reward_failure = -10000

        self.semantic_levels = semantic_levels
        self.gap_between_items = gap_between_items
        self.prop_target_absent = prop_target_absent
        self.length_observations = length_observations
        self.p_obs_len_cur = p_obs_len_cur
        self.p_obs_len_adj = p_obs_len_adj
        self.training = True
        self.n_item_lengths = 3
        
        self.logging = False
        self.log_session_variables = ["items", "target_present", "target_idx"]
        self.log_step_variables = [
            "duration_focus_ms",
            "duration_saccade_ms",
            "action_duration",
            "action",
            "rewards",
            "gaze_location",
            "path"
        ]
        self.set_variant()
        
    def seeding(self, seed=None):
        self.random_state = np.random.default_rng(seed)
        self.random_param = np.random.default_rng(seed)

    def set_variant(self, variant=None):
        if variant is None:
            variant = self.variant

        f_dur_arr = [2.0702, 1.7300, 2.8800, 2.8400]
        d_sel_arr = [0.00, 0.32, 0.30, 0.29]
        p_rec_arr = [0.00, 0.00, 0.87, 0.69]
        p_sem_arr = [0.00, 0.00, 0.00, 0.93]
        free_params = [f_dur_arr[variant], d_sel_arr[variant], p_rec_arr[variant], p_sem_arr[variant]]
        
        self.set_free_params(free_params)

    def set_free_params(self, fixed_params):
        param_min_arr = [0.00, 0.00, 0.00, 0.00]
        param_max_arr = [6.00, 1.00, 1.00, 1.00]
        param_labels = [
            "focus_duration_100ms",
            "selection_delay_s",
            "menu_recall_probability",
            "p_obs_adjacent",
        ]
        self.v = dict()
        for label, val in zip(param_labels, fixed_params):
            self.v[label] = val

        self.z = (-1) * np.ones((4,))
        for i, label in enumerate(param_labels):
            self.z[i] = (self.v[label] - param_min_arr[i]) / (param_max_arr[i] - param_min_arr[i]) * 2 - 1

    def sample_free_params(self):
        assert self.variable_params
        self.z = (-1) * np.ones((4,))
        self.z[:self.variant + 1] = self.random_param.uniform(low=-1., high=1., size=(self.variant + 1,))

        param_min_arr = [0.00, 0.00, 0.00, 0.00]
        param_max_arr = [6.00, 1.00, 1.00, 1.00]
        param_labels = [
            "focus_duration_100ms",
            "selection_delay_s",
            "menu_recall_probability",
            "p_obs_adjacent",
        ]
        for i, label in enumerate(param_labels):
            self.v[label] = (self.z[i] + 1) / 2 * (param_max_arr[i] - param_min_arr[i]) + param_min_arr[i]

    def clean(self):
        self.training_menus = list()

    def reset(self, fixed_variant=None, fixed_params=None):
        """ Called by the library to reset the state
        """
        # state hidden from agent
        self.items, self.target_present, self.target_idx = self._get_menu()

        # state observed by agent
        obs_items = [MenuItem(ItemRelevance.NOT_OBSERVED, ItemLength.NOT_OBSERVED) for i in range(self.n_items)]
        focus = Focus.ABOVE_MENU
        quit = False
        self.state = State(obs_items, focus, quit)
        self.prev_state = self.state.copy()

        # misc environment state variables
        self.action_duration = None
        self.duration_focus_ms = None
        self.duration_saccade_ms = None
        self.action = None
        self.gaze_location = None
        self.n_actions = 0
        self.item_locations = np.arange(
            self.gap_between_items,
            self.gap_between_items * (self.n_items + 2),
            self.gap_between_items
        )
        if self.logging:
            self._start_log_for_new_session()

        if self.variable_params:
            if fixed_params is not None:
                self.set_free_params(fixed_params)
            elif fixed_variant is not None:
                self.set_variant(variant=fixed_variant)
            else:
                self.sample_free_params()
            return np.append(self.state.arr(), self.z, axis=0)
        return self.state.arr()

    def step(self, action):
        """ Changes the state of the environment based on agent action """
        self.action = Action(action)
        self.prev_state = self.state.copy()
        self.state, self.duration_focus_ms, self.duration_saccade_ms = self.do_transition(self.state, self.action)
        self.action_duration = self.duration_focus_ms + self.duration_saccade_ms
        self.gaze_location = int(self.state.focus)
        self.n_actions += 1
        if self.logging:
            self._log_transition()
            
        if self.variable_params:
            return np.append(self.state.arr(), self.z, axis=0), self.get_reward(), self.is_finished(), dict()
        return self.state.arr(), self.get_reward(), self.is_finished(), dict()

    def get_reward(self):
        """ Returns the current reward based on the state of the environment
        """
        # this function should be deterministic and without side effects
        if self.has_found_item is True:
            # reward for finding target
            return self.reward_success + int(-1 * self.action_duration)
        elif self.has_quit is True:
            if self.target_present is False:
                # reward for quitting when target is absent
                return self.reward_success
            else:
                # penalty for quitting when target is present
                return self.reward_failure
        # default penalty for spending time
        return int(-1 * self.action_duration)

    def is_finished(self):
        """ Returns true when the task is in end state """
        # this function should be deterministic and without side effects
        if self.n_actions >= self.max_number_of_actions_per_session:
            return True
        elif self.has_found_item is True:
            return True
        elif self.has_quit is True:
            return True
        return False
        
    def getSensors(self):
        """ Returns a scalar (enumerated) measurement of the state """
        # this function should be deterministic and without side effects
        return [tuple(self.state.obs_items).__hash__()]  # needs to return a list

    @property
    def has_found_item(self):
        return self.state.focus != Focus.ABOVE_MENU and self.state.obs_items[int(self.state.focus)].item_relevance == ItemRelevance.TARGET_RELEVANCE

    @property
    def has_quit(self):
        return self.state.quit

    def _observe_relevance_at(self, state, focus):
        state.obs_items[focus].item_relevance = self.items[focus].item_relevance
        return state

    def _observe_length_at(self, state, focus):
        state.obs_items[focus].item_length = self.items[focus].item_length
        return state

    def do_transition(self, state, action):
        """ Changes the state of the environment based on agent action.
            Also depends on the unobserved state of the environment.

        Parameters
        ----------
        state : State
        action : Action

        Returns
        -------
        tuple (State, int) with new state and action duration in ms
        """
        state = state.copy()
        # menu recall event may happen at first action
        if self.n_actions == 0:
            if self.random_state.random() < float(self.v["menu_recall_probability"]):
                state.obs_items = [item.copy() for item in self.items]

        if action == Action.QUIT:
            state.quit = True
            focus_duration = 0
            saccade_duration = 0
        else:
            # saccade
            # item_locations are off-by-one to other lists
            if state.focus != Focus.ABOVE_MENU:
                amplitude = abs(self.item_locations[int(state.focus) + 1] - self.item_locations[int(action) + 1])
            else:
                amplitude = abs(self.item_locations[0] - self.item_locations[int(action) + 1])
            saccade_duration = int(37 + 2.7 * amplitude)
            state.focus = Focus(int(action))  # assume these match

            # fixation
            focus_duration = int(self.v["focus_duration_100ms"] * 100)
            # semantic observation at focus
            state = self._observe_relevance_at(state, int(state.focus))
            # possible length observations
            if self.length_observations is True:
                if int(state.focus) > 0 and self.random_state.random() < self.p_obs_len_adj:
                    state = self._observe_length_at(state, int(state.focus) - 1)
                if self.random_state.random() < self.p_obs_len_cur:
                    state = self._observe_length_at(state, int(state.focus))
                if int(state.focus) < self.n_items-1 and self.random_state.random() < self.p_obs_len_adj:
                    state = self._observe_length_at(state, int(state.focus) + 1)
            # possible semantic peripheral observations
            if int(state.focus) > 0 and self.random_state.random() < float(self.v["p_obs_adjacent"]):
                state = self._observe_relevance_at(state, int(state.focus) - 1)
            if int(state.focus) < self.n_items-1 and self.random_state.random() < float(self.v["p_obs_adjacent"]):
                state = self._observe_relevance_at(state, int(state.focus) + 1)

            # found target -> will click
            if state.focus != Focus.ABOVE_MENU and state.obs_items[int(state.focus)].item_relevance == ItemRelevance.TARGET_RELEVANCE:
                focus_duration += int(self.v["selection_delay_s"] * 1000)

        return state, focus_duration, saccade_duration

    def _get_menu(self):
        # generate menu item semantic relevances and lengths
        items, target_idx = self._get_semantic_menu(
            self.menu_groups,
            self.menu_items_per_group,
            self.semantic_levels,
            self.prop_target_absent,
            permutation=(self.menu_type != "semantic")
        )
        lengths = self.random_state.integers(self.n_item_lengths, size=len(items)).tolist()
        target_present = (target_idx != None)
        if target_present:
            items[target_idx].item_relevance = ItemRelevance.TARGET_RELEVANCE
            target_len = lengths[target_idx]
        else:
            # if target not present, choose target length randomly
            target_len = self.random_state.integers(self.n_item_lengths)

        for i, length in enumerate(lengths):
            if length == target_len:
                items[i].item_length = ItemLength.TARGET_LENGTH
            else:
                items[i].item_length = ItemLength.NOT_TARGET_LENGTH

        menu = (tuple(items), target_present, target_idx)
        return menu

    def _get_semantic_menu(self, n_groups, n_each_group, n_grids, p_absent, permutation=False):
        assert n_groups > 0
        assert n_each_group > 0
        assert n_grids > 0
        menu, target = self._semantic(n_groups, n_each_group, p_absent)
        if permutation:
            menu = self.random_state.permutation(menu)
        gridded_menu = self._griding(menu, target, n_grids)
        menu_length = n_each_group * n_groups

        start = 1 / float(2 * n_grids)
        stop = 1
        step = 1 / float(n_grids)
        grids = np.arange(start, stop, step)

        coded_menu = [MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED) for _ in range(menu_length)]
        for i, item in enumerate(gridded_menu):
            if not (item - grids[0]).any():
                coded_menu[i] = MenuItem(ItemRelevance.LOW_RELEVANCE, ItemLength.NOT_OBSERVED)
            elif not (item - grids[1]).any():
                coded_menu[i] = MenuItem(ItemRelevance.MED_RELEVANCE, ItemLength.NOT_OBSERVED)
            elif not (item - grids[2]).any():
                coded_menu[i] = MenuItem(ItemRelevance.HIGH_RELEVANCE, ItemLength.NOT_OBSERVED)
        return coded_menu, target

    def _semantic(self, n_groups, n_each_group, p_absent):
        n_items = n_groups * n_each_group
        target_value = 1

        semantic_menu = np.array([0] * n_items)[np.newaxis]

        """alpha and beta for target/relevant menu items"""
        target_group_parameters = [3.1625, 1.2766]

        """alpha and beta for non-target/irrelevant menu items"""
        non_target_group_paremeters = [5.3665, 18.8826]

        """alpha and beta parameters for the menus with no target"""
        absent_menu_parameters = [2.1422, 13.4426]

        """randomly select whether the target is present or abscent"""
        target_type = self.random_state.random()
        target_location = self.random_state.integers(n_items)

        if target_type > p_absent:
            target_group_samples = self.random_state.beta(
                *target_group_parameters,
                (n_each_group,)
            )
            distractor_group_samples = self.random_state.beta(
                *non_target_group_paremeters,
                (n_items,)
            )
            """ step 3 using the samples above to create Organised Menu and Random Menu
                and then add the target group
                the menu is created with all distractors first
            """
            semantic_menu = distractor_group_samples
            target_in_group = math.ceil((target_location + 1) / float(n_each_group))
            begin = (target_in_group - 1) * n_each_group
            end = (target_in_group - 1) * n_each_group + n_each_group

            semantic_menu[begin:end] = target_group_samples
            semantic_menu[target_location] = target_value
        else:
            target_location = None
            semantic_menu = self.random_state.beta(
                *absent_menu_parameters,
                (n_items,)
            )
        return semantic_menu, target_location
    
    def _griding(self, menu, target, n_levels):
        start = 1 / float(2 * n_levels)
        stop = 1
        step = 1 / float(n_levels)
        np_menu = np.array(menu)[np.newaxis]
        griding_semantic_levels = np.arange(start, stop, step)
        temp_levels = abs(griding_semantic_levels - np_menu.T)
        min_index = temp_levels.argmin(axis=-1)
        gridded_menu = griding_semantic_levels[min_index]
        if target is not None:
            gridded_menu[target] = 1
        return gridded_menu.T

    def start_logging(self):
        self.logging = True
        self.log = dict()

    def end_logging(self):
        self.logging = False
        self.log = None

    def _start_log_for_new_session(self):
        """ Set up log when new session starts
        """
        if self.log != None:
            if "session" not in self.log:
                self.log["session"] = 0
                self.log["sessions"] = [dict()]
            else:
                self.log["session"] += 1
                self.log["sessions"].append(dict())
            self.step_data = self.log["sessions"][self.log["session"]]
            for varname in self.log_session_variables:
                self.step_data[varname] = getattr(self, varname)
            for varname in self.log_step_variables:
                if varname == "path":
                    self.step_data["path"] = Path([])
                else:
                    self.step_data[varname] = list()

    def _log_transition(self):
        """ Should be called after transition
        """
        if self.log != None:
            for varname in self.log_step_variables:
                if varname == "rewards":
                    self.step_data["rewards"].append(self.get_reward())
                elif varname == "path":
                    self.step_data["path"].append(Transition(self.prev_state, self.action, self.state))
                else:
                    self.step_data[varname].append(getattr(self, varname))