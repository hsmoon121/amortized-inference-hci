"""
An implementation of "Point-and-click model"
from Seungwon et al., "A simulation model of intermittently controlled point-and-click behaviour." CHI 2021.

Original code: https://github.com/dodoseung/point-and-click-simulation-model
"""

import os.path
from pathlib import Path
import numpy as np
from scipy import interpolate
import csv

PATH_GAIN = os.path.join(Path(__file__).parent, "materials", "0.6875.txt")
PATH_ROTATION_MAP = os.path.join(Path(__file__).parent, "materials", "rotation_map_rad.csv")


### Mouse acceleration function
def rot_mat(x, y, rad):
    c, s = np.cos(rad), np.sin(rad)
    r = np.array(((c, -s), (s, c)))
    vec = np.array((x, y))
    mat = r.dot(vec)

    return mat[0], mat[1]


def mm2in(d):
    return d / 25.4


def in2m(d):
    return 0.0254 * d


def gain_func(vel):
    return f(vel)


def gain_func_can(vel):
    return g(vel)


cpi = 400
hz = 125
counts = []
pixels = []
with open(PATH_GAIN, "r") as file:
    for s in file:
        line = s.replace("\n", "").split(": ")
        counts.append(int(line[0]))
        pixels.append(float(line[1]))

motor_speed, gain, visual_speed = [], [], []
for c, p in zip(counts, pixels):
    ms = in2m(c / cpi) * hz
    vs = in2m(p / 110) * hz
    g = 0.0 if ms == 0.0 else vs / ms
    motor_speed.append(ms)
    gain.append(g)
    visual_speed.append(vs)

f = interpolate.interp1d(motor_speed, gain, fill_value="extrapolate")
g = interpolate.interp1d(visual_speed, gain, fill_value="extrapolate")


### Mouse coordinate disturbance
### Converting from hand movement to cursor location
with open(PATH_ROTATION_MAP, newline="") as file:
    rotation_map_rad = list(csv.reader(file))


def get_hand_orientation(hand_loc, forearm_length):
    # hand location in physical surface (when neutral position x,y=0)

    x = np.arange(-1.5, 1.51, 0.1)
    y = -np.arange(-1.5, 1.51, 0.1)
    f = interpolate.interp2d(x, y, rotation_map_rad, kind="cubic")
    # see surface_definition.png for more details on the definition of coordinate system
    hand_x = hand_loc[0]
    hand_y = hand_loc[1]

    normalized_hand_x = hand_x / forearm_length
    normalized_hand_y = hand_y / forearm_length

    # get hand orientation from rotation_map_rad.csv (matrix)
    # the matrix divides the surface into 31 by 31 grid, and for each grid point,
    # the matrix contains the expected hand orientation in rad (= the same as mouse rotation)
    # intermediate location should be interpolated from neighborhoods

    # rotation_map_rad[int(normalized_hand_x)][int(normalized_hand_y)]
    hand_orientation = f(normalized_hand_x, normalized_hand_y)

    return float(hand_orientation)


def get_cursor_displacement(hand_start_loc, hand_end_loc, user_forearm_length, gain):

    # hand_start_loc, hand_end_loc = x,y location of hand on the mouse pad (a predefined surface)
    # hand_start_angle, hand_end_angle = upright angle as zero rad, clock-wise positive
    # gain = value of gain function (now simply assumed to be constant)

    current_hand_orientation = get_hand_orientation(hand_end_loc, user_forearm_length)
    prev_hand_orientation = get_hand_orientation(hand_start_loc, user_forearm_length)

    hand_displacement = np.linalg.norm(
        np.array(hand_end_loc) - np.array(hand_start_loc)
    )
    # net_hand_rotation = prev_hand_orientation - current_hand_orientation
    net_hand_rotation = current_hand_orientation - prev_hand_orientation
    if net_hand_rotation == 0:
        net_hand_rotation = np.finfo(float).tiny

    # get cursor displacement
    cursor_displacement = (
        gain * hand_displacement * 2 * np.sin(net_hand_rotation / 2)
    ) / net_hand_rotation

    # normalized hand velocity vector
    hand_displacement = (
        np.finfo(float).tiny if hand_displacement == 0 else hand_displacement
    )
    hand_dx = (hand_end_loc[0] - hand_start_loc[0]) / hand_displacement
    hand_dy = (hand_end_loc[1] - hand_start_loc[1]) / hand_displacement

    # resulting cursor dx and dy after passing through the mouse disturbance effect
    cursor_dx = (
        hand_dx * np.cos(net_hand_rotation / 2 + prev_hand_orientation)
        - hand_dy * np.sin(net_hand_rotation / 2 + prev_hand_orientation)
    ) * cursor_displacement
    cursor_dy = (
        hand_dx * np.sin(net_hand_rotation / 2 + prev_hand_orientation)
        + hand_dy * np.cos(net_hand_rotation / 2 + prev_hand_orientation)
    ) * cursor_displacement

    return cursor_dx, cursor_dy
