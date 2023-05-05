"""
An implementation of "Point-and-click model"
from Do et al., "A simulation model of intermittently controlled point-and-click behaviour." CHI 2021.

Original code: https://github.com/dodoseung/point-and-click-simulation-model
"""

import numpy as np
from .model_config import CONFIG


def w_intersect(cursor_pos, cursor_vel, target_pos, target_vel, target_radius):
    target_radius = target_radius[0]
    cursor_pos_end = cursor_pos + (cursor_vel - target_vel)

    nominator = np.linalg.norm(
        np.cross(cursor_pos_end - cursor_pos, cursor_pos - target_pos)
    )
    denominator = np.linalg.norm(cursor_pos_end - cursor_pos)
    center_to_line = nominator / denominator

    dist_cursor_target = np.linalg.norm(target_pos - cursor_pos)

    if dist_cursor_target <= target_radius:
        value = np.sqrt(target_radius ** 2 - center_to_line ** 2)
        bot = np.sqrt(dist_cursor_target ** 2 - center_to_line ** 2)
        vec1 = cursor_pos_end - cursor_pos
        vec2 = cursor_pos - target_pos
        if vec1.dot(vec2) > 0:
            return value - bot
        else:
            return value + bot

    if center_to_line <= target_radius:
        value = np.sqrt(target_radius ** 2 - center_to_line ** 2)
        w_intersect_value = 2 * value
    else:
        w_intersect_value = 0

    return w_intersect_value


def w_t(w_intersect_value, cursor_vel, target_vel):
    vel_relative = cursor_vel - np.array(target_vel)
    wt = w_intersect_value / np.linalg.norm(vel_relative)

    return wt


def mean(c_mu, wt):
    click_timing_mean = c_mu * wt

    return click_timing_mean


def variance(c_sigma, _p, _nu, tc, _delta):
    nominator = (c_sigma ** 2) * (_p ** 2)
    denominator = 1 + (_p / (1 / (np.exp(_nu * tc) - 1) + _delta)) ** 2
    click_timing_variance = np.sqrt(nominator / denominator)

    return click_timing_variance


# The click timing
def model(
    cursor_pos_init,
    cursor_vel,
    target_pos,
    target_vel,
    target_radius,
    tc,
    user_params,
):
    w_intersect_value = w_intersect(
        cursor_pos_init, cursor_vel, target_pos, target_vel, target_radius
    )
    w_t_value = w_t(w_intersect_value, cursor_vel, target_vel)
    tc -= w_t_value / 2
    click_timing_mean = mean(user_params["cmu"], w_t_value)
    click_timing_variance = variance(
        user_params["csigma"], CONFIG["p"], user_params["nu"], tc, user_params["delta"]
    )
    click_timing = np.random.normal(click_timing_mean, click_timing_variance, 1)
    # print(f"click_timing: {click_timing[0]:.3f}, mean: {click_timing_mean:.3f}, var: {click_timing_variance:.3f}")
    return click_timing[0]


def index_of_difficulty(
    cursor_pos_init,
    cursor_vel,
    target_pos,
    target_vel,
    target_radius,
    tc,
    _nu,
    _delta,
    _p,
):
    w_intersect_value = w_intersect(
        cursor_pos_init, cursor_vel, target_pos, target_vel, target_radius
    )
    w_t_value = w_t(w_intersect_value, cursor_vel, target_vel)

    tc -= w_t_value / 2
    nominator = _p
    denominator = np.sqrt(1 + (_p / (1 / (np.exp(_nu * tc) - 1) + _delta)) ** 2)
    d_t_value = nominator / denominator

    if w_t_value == 0:
        w_t_value = np.finfo(float).tiny

    idx_of_diff = np.log2(d_t_value / w_t_value)

    return idx_of_diff


def total_click_time(
    click_timing,
    tc,
    cursor_pos,
    cursor_vel,
    target_pos,
    target_vel,
    target_radius,
):
    w_intersect_value = w_intersect(
        cursor_pos, cursor_vel, target_pos, target_vel, target_radius
    )
    w_t_value = w_t(w_intersect_value, cursor_vel, target_vel)

    time_enter = tc - (w_t_value / 2)
    time_total = time_enter + click_timing

    if time_enter < 0 and click_timing >= tc:
        time_total = tc - 0.00001
    elif time_total < 0:
        time_total = 0
    elif w_t_value == 0:
        time_total = tc - 0.00001
    elif time_total > tc:
        time_total = tc - 0.00001
    # print(f"time_total: {time_total:.3f}, tc: {tc:.3f}, wt: {w_t_value:.3f}")
    return time_total