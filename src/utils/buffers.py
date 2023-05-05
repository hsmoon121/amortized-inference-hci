import time
import numpy as np
import pandas as pd
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity, use_traj=True):
        self.capacity = capacity
        self._buffer = {
            "params": pd.Series([None,] * capacity),
            "stat_data": pd.Series([None] * capacity),
        }
        self.use_traj = use_traj
        if use_traj:
            self._buffer["traj_data"] = pd.Series([None] * capacity)

        self._push_time = deque(maxlen=capacity // 10)
        self._clear()

    def push(self, params, stat_data, traj_data=None):
        push_size = stat_data.shape[0]
        for i in range(push_size):
            start_t = time.time()
            if self.use_traj:
                self._add_with_index(self._pos, params[i], stat_data[i], traj_data[i])
            else:
                self._add_with_index(self._pos, params[i], stat_data[i])
            self._update()
            self._push_time.append(time.time() - start_t)
        
    def sample(self, batch_size):
        indices = np.random.choice(self._size, batch_size)
        return self._sample_with_indices(indices)

    def get_avg_push_time(self):
        return sum(self._push_time) / len(self._push_time)

    def _add_with_index(self, index, params, stat_data, traj_data=None):
        self._buffer["params"][index] = params
        self._buffer["stat_data"][index] = stat_data
        if self.use_traj:
            self._buffer["traj_data"][index] = traj_data

    def _sample_with_indices(self, indices):
        if self.use_traj:
            return (
                np.array(list(self._buffer["params"][indices]), dtype=np.float32),
                np.array(list(self._buffer["stat_data"][indices]), dtype=np.float32),
                list(self._buffer["traj_data"][indices]),
            )
        return (
            np.array(list(self._buffer["params"][indices]), dtype=np.float32),
            np.array(list(self._buffer["stat_data"][indices]), dtype=np.float32),
        )

    def _update(self):
        self._pos = (self._pos + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def _clear(self):
        self._pos = 0
        self._size = 0

    def __len__(self):
        return self._size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Deprecated
    """
    def __init__(self, capacity, use_traj=True, alpha=0.1, beta=0.4):
        super().__init__(capacity, use_traj)
        self.alpha = alpha
        self.beta = beta
        self._priorities = np.zeros((self.capacity,), dtype=np.float32)

    def push(self, params, stat_data, traj_data=None):
        push_size = len(stat_data) if isinstance(stat_data, list) else 1
        for i in range(push_size):
            start_t = time.time()
            if not isinstance(stat_data, list):
                self._add_with_index(self._pos, params, stat_data, traj_data)
            elif self.use_traj:
                self._add_with_index(self._pos, params[i], stat_data[i], traj_data[i])
            else:
                self._add_with_index(self._pos, params[i], stat_data[i])
            max_prio = self._priorities.max() if self._size > 0 else 1.0
            self._priorities[self._pos] = max_prio
            self._update()
            self._push_time.append(time.time() - start_t)

    def sample(self, batch_size):
        prios = self._priorities[:self._size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self._size, batch_size, p=probs)

        weights = (self._size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        encoded_sample = self._sample_with_indices(indices)
        return tuple([indices, weights] + list(encoded_sample))
        
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self._priorities[idx] = prio
    
