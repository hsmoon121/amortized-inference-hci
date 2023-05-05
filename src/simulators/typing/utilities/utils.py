"""
An implementation of "Touchscreen typing model"
from Jokinen et al., "Touchscreen typing as optimal supervisory control." CHI 2021.

Original code: https://github.com/aditya02acharya/TypingAgent
"""

from math import sqrt, atan, pi, log, exp
from typing import Union
import numpy as np
import torch as th


def distance(origin, destination):
    """
    Euclidian distance between two points.

    :param origin: array of size 2 representing (x,y) coordinate of a point.
    :param destination: array of size 2 representing (x,y) coordinate of a point.
    :return: distance: float distance between given points.
    """

    return sqrt(((origin[0] - destination[0]) ** 2 + (origin[1] - destination[1]) ** 2))


def visual_distance(dist, user_distance):
    """
    Calculate visual distance, as degrees, given euclidian distance.
    User distance needs to be given in the same unit.

    :param dist: euclidian distance between points.
    :param user_distance: distance of user from the device.
    :return:
    """
    return 180 * (atan(dist / user_distance) / pi)


def EMMA_fixation_time(dist, freq=0.1, t_prep=0.135):
    """
    Mathematical model for saccade duration from EMMA (Salvucci, 2001).

    :param dist: eccentricity in visual angle.
    :param freq: frequency of object being encoded. Value in (0,1).
    :return: EMMA_breakdown : tuple containing (preparation_time, execution_time, left_encoding_time).
    :return: total_time : total eye movement time.
    :return: moved : true if encoding time > preparation time. false otherwise.
    """

    # EMMA parameters
    emma_K = 0.006
    emma_k = 0.4
    emma_exec = 0.07
    emma_saccade = 0.002
    # t_prep = 0.135 # <-- default values for t_prep

    # visual encoding time
    t_enc = emma_K * -log(freq) * exp(emma_k * dist)

    # if encoding time < movement preparation time then no movement
    if t_enc < t_prep:
        return (t_enc, 0, 0), t_enc, False

    # movement execution time
    t_exec = emma_exec + emma_saccade * dist
    # eye movement time (preparation time + execition time)
    t_sacc = t_prep + t_exec

    # if encoding time less then movement time
    if t_enc <= t_sacc:
        return (t_prep, t_exec, 0), t_sacc, True

    # if encoding left after movement time
    e_new = (emma_k * -log(freq))
    t_enc_new = (1 - (t_sacc / t_enc)) * e_new

    return (t_prep, t_exec, t_enc_new), t_sacc + t_enc_new, True


def WHo_mt(dist, sigma, alpha=0.6, k=0.12):
    """
    Speed Accuracy model for generating finger movement time.

    :param dist: euclidian distance between points.
    :param sigma: speed-accuracy trade-off variance.
    :return: mt: movement time.
    """
    x0 = 0.092
    y0 = 0.0018
    x_min = 0.006
    x_max = 0.06
    # alpha = 0.6 # <-- Speed-accuracy bias
    # k = 0.08 # <-- Total resource

    if dist == 0:
        dist = 0.0000001

    mt = pow((k * pow(((sigma - y0) / dist), (alpha - 1))), 1 / alpha) + x0

    return mt


def parse_transition_index(record):
    """
    Utility function to format transition function index.
    :param record: tuple as string.
    :return: list of features.
    """
    token = record[1:-1].split(', ')
    token[0] = int(token[0])
    token[1] = float(token[1])
    token[2] = (int(token[2][1:]), int(token[3][:-1]))
    return token


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    It supports 'cpu', 'cuda', 'mps'
    By default, it tries to use the gpu ('cuda' or 'mps').
    :param device: One for 'auto', 'cuda', 'cpu', 'mps'
    :return:
    """
    # Cuda by default
    if device == "auto":
        if float(th.__version__[:4]) < 1.13:
            if th.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            # if th.backends.mps.is_available():
            #     device = "mps"
            if th.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            
    return th.device(device)
