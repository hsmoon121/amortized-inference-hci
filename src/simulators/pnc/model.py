"""
An implementation of "Point-and-click model"
from Seungwon et al., "A simulation model of intermittently controlled point-and-click behaviour." CHI 2021.

Original code: https://github.com/dodoseung/point-and-click-simulation-model
"""

import numpy as np

from . import click_action_module as click
from . import motor_control_module as motor
from . import visual_perception_module as visual
from . import mouse_module as mouse
from . import upper_limb_module as limb
from .model_config import CONFIG


def model(state_true, state_cog, action, user_params):

    # States
    cursor_pos_x, cursor_pos_y, cursor_vel_x, cursor_vel_y = (
        state_cog["cp"][0],
        state_cog["cp"][1],
        state_cog["cv"][0],
        state_cog["cv"][1],
    )

    (
        c_true_pos_x,
        c_true_pos_y,
        target_pos_x,
        target_pos_y,
        target_vel_x,
        target_vel_y,
        target_radius,
        h_pos_x,
        h_pos_y,
    ) = (
        state_true["cp"][0],
        state_true["cp"][1],
        state_true["tp"][0],
        state_true["tp"][1],
        state_true["tv"][0],
        state_true["tv"][1],
        state_true["tr"],
        state_true["hp"][0],
        state_true["hp"][1],
    )

    click_decision = CONFIG["K"][action // len(CONFIG["Th"])]
    tp = round(CONFIG["Tp"] / CONFIG["INTERVAL"])
    th = round(CONFIG["Th"][action % len(CONFIG["Th"])] * user_params["max_th"] / CONFIG["max_th"] / CONFIG["INTERVAL"])
    th = max(tp, th)

    # Default click timing
    time_total = 99

    target_true = [target_pos_x, target_pos_y, target_vel_x, target_vel_y]

    # Target information at SA
    # Target information with the visual noise
    # Target information at RP
    tar_pos, tar_vel = visual.add_noise_past(
        tp,
        np.array([target_pos_x, target_pos_y]),
        np.array([target_vel_x, target_vel_y]),
        target_radius,
        user_params["sigmav"]
    )
    target_info = np.array([tar_pos[0], tar_pos[1], tar_vel[0], tar_vel[1]])
    # Predicted target information after Th
    tar_pos, tar_vel = motor.boundary(th, tar_pos, tar_vel, target_radius)

    # Set the prediction horizon
    pred_horizon = th if click_decision else tp

    # otg_ideal[x/y, pos/vel, length]
    otg_ideal_x = motor.otg_new(
        state_cog["cp"][0],
        state_cog["cv"][0],
        tar_pos[0],
        tar_vel[0],
        th,
        pred_horizon,
    )

    otg_ideal_y = motor.otg_new(
        state_cog["cp"][1],
        state_cog["cv"][1],
        tar_pos[1],
        tar_vel[1],
        th,
        pred_horizon,
    )

    otg_ideal = np.stack([np.array(otg_ideal_x), np.array(otg_ideal_y)])

    botg_ideal = otg_ideal.clip(
        min=np.array([np.finfo(float).tiny, -np.finfo(np.float32).max])[None, :, None],
        max=np.array(
            [
                [CONFIG["WINDOW_WIDTH"], np.finfo(np.float32).max],
                [CONFIG["WINDOW_HEIGHT"], np.finfo(np.float32).max],
            ]
        )[:, :, None],
    )

    if np.any(botg_ideal != otg_ideal):
        mask = botg_ideal != otg_ideal
        botg_ideal *= 1 - np.roll(mask, 1, axis=1)

    # otg_ideal[:, :, :pred_horizon] = botg_ideal[:, :, :pred_horizon]
    otg_ideal[:, :, :2] = botg_ideal[:, :, :2]  # following possible bug

    # Get the position, velocity, and acceleration information of the cursor
    c_vel = otg_ideal[:, 1, : pred_horizon + 1].copy()

    # Set the mouse noise
    mouse_gain = CONFIG["MOUSE_GAIN"]
    # mouse_gain = mouse.gain_func_can((c_vel ** 2).sum(axis=0) ** 0.5)
    # for idx in range(pred_horizon + 1):
    #     if mouse_gain[idx] == 0:
    #         mouse_gain[idx] = np.finfo(float).tiny
    # print(mouse_gain)

    # Get the acceleration
    acc = ((c_vel / mouse_gain)[:, 1:] - (c_vel / mouse_gain)[:, :-1]) / CONFIG[
        "INTERVAL"
    ]

    # Get the ideal hand pos
    _, _, h_pos_x_ideal, h_pos_y_ideal = limb.mouse_noise(
        c_vel[0] / mouse_gain,
        c_vel[1] / mouse_gain,
        h_pos_x,
        h_pos_y,
        mouse_gain,
    )

    h_pos_delta_x = h_pos_x_ideal - h_pos_x
    h_pos_delta_y = h_pos_y_ideal - h_pos_y

    # Set the motor noise
    nc = np.array([user_params["nv"], user_params["np"]])
    _, _, c_vel_x, c_vel_y = limb.motor_noise(
        c_vel[0] / mouse_gain, c_vel[1] / mouse_gain, pred_horizon, 0, 0, nc
    )

    # Set the mouse noise
    pos_dx_mouse, pos_dy_mouse, h_pos_x, h_pos_y = limb.mouse_noise(
        c_vel_x.copy(),
        c_vel_y.copy(),
        h_pos_x,
        h_pos_y,
        mouse_gain,
    )

    c_pos_dx = (
        ((c_vel_x * mouse_gain)[1:] + (c_vel_x * mouse_gain)[:-1])
        / 2
        * CONFIG["INTERVAL"]
    )
    c_pos_dy = (
        ((c_vel_y * mouse_gain)[1:] + (c_vel_y * mouse_gain)[:-1])
        / 2
        * CONFIG["INTERVAL"]
    )
    vel_mouse_noise_x = (sum(pos_dx_mouse) - sum(c_pos_dx)) / (
        pred_horizon * CONFIG["INTERVAL"]
    )
    vel_mouse_noise_y = (sum(pos_dy_mouse) - sum(c_pos_dy)) / (
        pred_horizon * CONFIG["INTERVAL"]
    )
    c_pos_dx, c_pos_dy, c_vel_x, c_vel_y = limb.motor_noise(
        c_vel_x * mouse_gain,
        c_vel_y * mouse_gain,
        pred_horizon,
        vel_mouse_noise_x,
        vel_mouse_noise_y,
        [0, 0],
    )

    # Ideal last executed cursor position and velocity for next BUMP
    c_pos_delta = otg_ideal[:, 0, pred_horizon] - otg_ideal[:, 0, 0]

    # Ideal cursor velocity
    c_vel_ideal = otg_ideal[:, 1, pred_horizon]

    # For Click Timing
    # Get the click timing
    if click_decision:
        target_next_pos, _ = motor.boundary(
            th, np.array(target_true[0:2]), np.array(target_true[2:4]), target_radius
        )

        tc = th * CONFIG["INTERVAL"]
        cursor_pos = np.array([c_true_pos_x, c_true_pos_y])
        cursor_vel = np.array([np.sum(c_pos_dx) / tc, np.sum(c_pos_dy) / tc])
        target_pos = np.array([target_true[0], target_true[1]])
        target_vel = np.array(
            [
                (target_next_pos[0] - target_true[0]) / tc,
                (target_next_pos[1] - target_true[1]) / tc,
            ]
        )

        click_timing = click.model(
            cursor_pos,
            cursor_vel,
            target_pos,
            target_vel,
            target_radius,
            tc,
            user_params,
        )
        time_total = click.total_click_time(
            click_timing,
            tc,
            cursor_pos,
            cursor_vel,
            target_pos,
            target_vel,
            target_radius,
        )

    # Acceleration
    acc_sum = sum((acc ** 2).sum(axis=0) ** 0.5)

    # Correct the trajectory which is out of the bound
    for idx in range(len(c_pos_dx)):
        if c_true_pos_x + sum(c_pos_dx[: idx + 1]) <= 0:
            c_pos_dx[idx] = -(c_true_pos_x + sum(c_pos_dx[:idx]))
            c_vel_x[idx + 1] = 0
        elif c_true_pos_x + sum(c_pos_dx[: idx + 1]) >= CONFIG["WINDOW_WIDTH"]:
            c_pos_dx[idx] = CONFIG["WINDOW_WIDTH"] - (
                c_true_pos_x + sum(c_pos_dx[:idx])
            )
            c_vel_x[idx + 1] = 0

        if c_true_pos_y + sum(c_pos_dy[: idx + 1]) <= 0:
            c_pos_dy[idx] = -(c_true_pos_y + sum(c_pos_dy[:idx]))
            c_vel_y[idx + 1] = 0
        elif c_true_pos_y + sum(c_pos_dy[: idx + 1]) >= CONFIG["WINDOW_HEIGHT"]:
            c_pos_dy[idx] = CONFIG["WINDOW_HEIGHT"] - (
                c_true_pos_y + sum(c_pos_dy[:idx])
            )
            c_vel_y[idx + 1] = 0

    # Summary of the outputs
    cursor_info = (c_pos_dx, c_pos_dy, c_vel_x, c_vel_y, c_pos_delta, c_vel_ideal)
    hand_info = h_pos_x, h_pos_y, h_pos_delta_x, h_pos_delta_y
    click_info = time_total
    effort_info = acc_sum

    return cursor_info, target_info, hand_info, click_info, effort_info
