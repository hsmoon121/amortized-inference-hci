"""
An implementation of "Menu search model"
from Kangasrääsiö et al., "Inferring cognitive models from data using approximate Bayesian computation." CHI 2017.

Original code: https://github.com/akangasr/cogsciabc
"""

import os, datetime
from collections import deque
import logging
logger = logging.getLogger(__name__)
from torch.utils.tensorboard import SummaryWriter


class Path():
    def __init__(self, transitions):
        self.transitions = transitions

    def append(self, transition):
        self.transitions.append(transition)

    def get_start_state(self):
        if len(self) < 1:
            raise ValueError("Path contains no transitions and thus no start state")
        return self.transitions[0].prev_state

    def __eq__(a, b):
        if type(a) != type(b):
            return False
        if len(a) != len(b):
            return False
        for t1, t2 in zip(a.transitions, b.transitions):
            if t1 != t2:
                return False
        return True

    def __len__(self):
        return len(self.transitions)

    def __repr__(self):
        ret = list()
        for t in self.transitions:
            ret.append("{};".format(t))
        return "".join(ret)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Path([transition.copy() for transition in self.transitions])

        
class Transition():
    def __init__(self, prev_state, action, next_state):
        self.prev_state = prev_state
        self.action = action
        self.next_state = next_state

    def __eq__(a, b):
        return a.__hash__() == b.__hash__()

    def __hash__(self):
        return (self.prev_state, self.action, self.next_state).__hash__()

    def __repr__(self):
        return "T({}+{}->{})".format(self.prev_state, self.action, self.next_state)

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return Transition(self.prev_state.copy(), self.action, self.next_state.copy())


class Logger:
    def __init__(self, name, window, write_board=False, board_path="./data/board/"):
        self.name = name
        self.scores = deque(maxlen=window)
        self.loss = deque(maxlen=window)
        self.avg_num = window
        self.board_path = board_path

        if write_board:
            self._set_tensorboard_writer()
        else:
            self.writer = None

    def _set_tensorboard_writer(self):
        os.makedirs(f"{self.board_path}/{self.name}", exist_ok=True)
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.writer = SummaryWriter(f"{self.board_path}/{self.name}/{datetime_str}")

    def is_empty(self):
        return not len(self.scores)

    def push(self, score, loss):
        self.scores.append(score)
        self.loss.append(loss)

    def clear(self):
        self.scores.clear()
        self.loss.clear()

    def _moving_avg(self):
        avg_score = sum(self.scores) / len(self.scores)
        avg_loss = sum(self.loss) / len(self.loss)
        return avg_score, avg_loss

    def log(self, ep, step, print_log=False, **kwargs):
        avg_score, avg_loss = self._moving_avg()
        self.write_scalar(
            step,
            score=float(avg_score),
            loss=float(avg_loss),
            **kwargs
        )
        if print_log:
            additional_log = ""
            for key, value in kwargs.items():
                additional_log += f", {key}: {value:.3f}"
            print(
                f"Episode: {ep}, Reward: {avg_score:.4f}, Loss: {avg_loss:.4f}"
                + additional_log
            )

    def write_scalar(self, step, **kwargs):
        if self.writer is not None:
            for key, value in kwargs.items():
                self.writer.add_scalar(key, value, step)