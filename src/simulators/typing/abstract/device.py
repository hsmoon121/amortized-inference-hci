"""
An implementation of "Touchscreen typing model"
from Jokinen et al., "Touchscreen typing as optimal supervisory control." CHI 2021.

Original code: https://github.com/aditya02acharya/TypingAgent
"""

import abc
import numpy as np
import os
from pathlib import Path

import logging


class Device(abc.ABC):
    def __init__(self):
        self.logger = self.logger = logging.getLogger(__name__)
        self.layout = None

    def load_layout(self, layout_config):
        layout_path = os.path.join(Path(__file__).parent.parent, "layouts")
        if os.path.exists(os.path.join(layout_path, layout_config)):
            self.layout = np.load(os.path.join(layout_path, layout_config))
            self.logger.info('layout loading completed - {%s}.' % layout_config)
        else:
            self.logger.error('failed to load layout file {%s}.' % layout_config)
