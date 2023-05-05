import os
from pathlib import Path
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from ..configs import default_pnc_config
from ..nets.dqn import EmbeddingDQN
from .pnc import model as pnc_model
from .pnc import motor_control_module as motor
from .pnc import visual_perception_module as visual
from .pnc.model_config import CONFIG, STD_PARAMS, MEAN_PARAMS, pnc_qnet_config


class PnCSimulator(object):
    """
    A simulator of point-and-click behavior
    Point-and-click model from CHI 2021 paper by Do et al.
    """
    def __init__(self, config=None, qnet=None):
        if config is None:
            config = deepcopy(default_pnc_config["simulator"])

        self.seeding(config["seed"])
        self.targeted_params = config["targeted_params"]
        self.prior = config["prior"]
        self.max_time = 5
        self.min_time = 0.15
        self.max_dist = 5 * 0.024
        if qnet is None:
            self._load_default_qnet()
        else:
            self.qnet = qnet
        self.free_params = deepcopy(MEAN_PARAMS)
        self.free_params["max_th"] = 2.5

    def _load_default_qnet(self):
        qnet_config = deepcopy(pnc_qnet_config)
        if "max_th" not in self.targeted_params:
            qnet_config["name"] = "pnc_dqn_without_th"
            qnet_config["z_size"] = 3
        else:
            qnet_config["name"] = "pnc_dqn"
        self.qnet = EmbeddingDQN(
            qnet_config,
            name=qnet_config["name"],
            task_name="pnc",
            model_path=os.path.join(Path(__file__).parent, "pnc", "models")
        )
        self.qnet.load()

    def simulate(
        self,
        n_param=1,
        sim_per_param=1,
        fixed_params=None,
        fixed_initial_cond=None,
        verbose=False
    ):
        """
        Simulates human behavior based on given (or sampled) free parameters

        Arguments (Inputs):
        - n_param: no. of parameter sets to sample to simulate (only used when fixed_params is not given)
        - sim_per_param: no. of simulation per parameter set
        - fixed_params: ndarray of free parameter (see below) sets to simulate
        =======
        Free params in point-and-click model (from CHI'22 paper)
        1) sigma_v: log-uniform (min=0.069, max=0.415)
        2) n_v:     log-uniform (min=0.145, max=0.413)
        3) c_sigma: log-uniform (min=0.055, max=0.400)
        4) T_h,max: uniform (min=0.5, max=2.5)
        =======
        - fixed_initial_cond: fix task environment (e.g., initial pos & vel for target & cursor)

        Outputs:
        - lognorm_params: free parameter sets (with log-normalized values) used for simulation
            > ndarray with size ((n_param), (dim. of free parameters))
        - stats: static (fixed-size) behavioral outputs for every trial (see below)
            > ndarray with size ((n_param), (sim_per_param), (dim. of static behavior))
        =======
        Static behavioral output (normalized)
        1) click success
        2) completion time
        3) click distance
        4) initial cursor position (2D)
        5) initial cursor velocity (2D)
        6) initial target position (2D)
        7) initial target velocity (2D)
        8) target radius
        =======
        - trajs: trajectory (variable-size) for every trial (see below)
        =======
        Data for each timstep in trajectory data (normalized)
        1) time difference from prev. timestep
        2) cursor position (2D)
        3) target position (2D)
        =======
        """
        if fixed_params is not None:
            lognorm_params = self._compute_lognorm(fixed_params, normalized=False)
            n_param = lognorm_params.shape[0]
        else:
            lognorm_params = self._sample_params(n_param)

        stats, trajs = [], []
        for i in (tqdm(range(n_param)) if verbose else range(n_param)):
            self._z = lognorm_params[i]
            self._set_free_params()

            if sim_per_param == 1:
                time = 0
                while time < (self.min_time / self.max_time):
                    stat, traj = self._simulate_trial(fixed_initial_cond)
                    time = stat[1]
                stats.append(stat)
                trajs.append(traj)
            else:
                stats_per_param, trajs_per_param = [], []
                for sim_i in range(sim_per_param):
                    time = 0
                    while time < (self.min_time / self.max_time):
                        if fixed_initial_cond is not None:
                            if fixed_initial_cond.ndim > 1:
                                assert fixed_initial_cond.shape[0] == sim_per_param
                                stat, traj = self._simulate_trial(fixed_initial_cond[sim_i])
                            else:
                                stat, traj = self._simulate_trial(fixed_initial_cond)
                        else:
                            stat, traj = self._simulate_trial(fixed_initial_cond)
                        time = stat[1]
                    stats_per_param.append(stat)
                    trajs_per_param.append(traj)

                stats.append(stats_per_param)
                trajs.append(trajs_per_param)
                
        if "nv" not in self.targeted_params:
            lognorm_params = lognorm_params[:, 1:]
                        
        return lognorm_params, np.array(stats, dtype=np.float32), trajs

    def _sample_params(self, n_param):
        sampled_param = self.param_random.uniform(
            low=-1,
            high=1,
            size=(n_param, len(self.targeted_params)),
        )
        if self.prior == "uniform":
            lognorm_params = self._compute_lognorm(sampled_param, normalized=True)
        elif self.prior == "log-uniform":
            lognorm_params = sampled_param
        else:
            raise ValueError(f"wrong simulator prior : {self.prior}")
            
        if "nv" not in self.targeted_params:
            lognorm_params = np.insert(lognorm_params, [0], [0], axis=1)
        return lognorm_params

    def convert_from_output(self, outputs):
        """
        outputs : (batch, param_sz) or (param_sz,) output results (log-normalized free params)
        """
        param_in = deepcopy(outputs)
        if len(np.array(param_in).shape) == 1:
            param_in = np.expand_dims(param_in, axis=0)

        param_out = np.zeros_like((param_in))
        for i, k in enumerate(self.targeted_params):
            if not k in self.free_params:
                raise Exception(f"{k} not in free_params")
            if k == "max_th":
                # As an exception, "max_th" is sampled from uniform range of [0.5, 2.5]
                param_out[:, i] = param_in[:, i] + 1.5
            else:
                scale = 1 + 3 * STD_PARAMS[k] / MEAN_PARAMS[k]
                param_out[:, i] = MEAN_PARAMS[k] * (scale ** param_in[:, i])
        return param_out

    def _compute_lognorm(self, param_in, normalized=True):
        """
        param_in : (batch, param_sz) or (param_sz,) free parameters
        normalized : whether the params are normlized (ranging from [-1, 1]) 
        """
        param_in = deepcopy(param_in)
        if len(np.array(param_in).shape) == 1:
            param_in = np.expand_dims(param_in, axis=0)

        target = self.targeted_params
        # In case we try to infer "max_th", but given gt param doesn't include "max_th"
        if ("max_th" in self.targeted_params) and (param_in.shape[-1] == len(self.targeted_params) - 1):
            param_in = np.concatenate((param_in, 1.5 * np.ones((param_in.shape[0], 1))), axis=-1)

        param_out = np.zeros_like((param_in))
        for i, k in enumerate(target):
            if not k in self.free_params:
                raise Exception(f"{k} not in free_params")
            if k == "max_th":
                # As an exception, "max_th" is sampled from uniform range of [0.5, 2.5]
                if normalized:
                    param_out[:, i] = param_in[:, i]
                else:
                    param_out[:, i] = param_in[:, i] - 1.5
            else:
                scale = 1 + 3 * STD_PARAMS[k] / MEAN_PARAMS[k]
                if normalized:
                    # normalized param_in: ranging from [-1, 1]
                    min = 1 / scale
                    max = scale
                    actual_scale = (max - min) * (param_in[:, i] + 1) / 2 + min
                else:
                    # param_in: ranging from [mean/scale, mean*scale]
                    actual_scale = param_in[:, i] / MEAN_PARAMS[k]
                # param_out (log-normalized): ranging from [-1, 1]
                param_out[:, i] = np.log(actual_scale) / np.log(scale)
        return param_out

    def _set_free_params(self):
        for i, k in enumerate(self.targeted_params):
            if not k in self.free_params:
                raise Exception(f"{k} not in free_params")
            if k == "max_th":
                # As an exception, "max_th" is sampled from uniform range of [0.5, 2.5]
                self.free_params[k] = self._z[i] + 1.5
            else:
                scale_base = 1 + 3 * STD_PARAMS[k] / MEAN_PARAMS[k]
                self.free_params[k] = MEAN_PARAMS[k] * (scale_base ** self._z[i])
            if k == "nv":
                self.free_params["np"] = self.free_params["nv"] * MEAN_PARAMS["np"] / MEAN_PARAMS["nv"]

    def _simulate_trial(self, fixed_initial_cond=None):
        idx, time, click_success = 0, 0, 0
        initial_cond = self._sample_initial_condition(fixed_initial_cond)
        traj = list([
            [0,] + list(self.tstate["cp"]) + list(self.tstate["tp"]),
        ])
        done = False
        while not done:
            if idx == 0: # new bump starts
                obs = self._observation()
                act = self._get_action(obs)
                click_time = self._set_bump(act)
            clicked = idx < (click_time / CONFIG["INTERVAL"]) < idx + 1

            if not clicked:
                self._tstate_step(idx)
                time += CONFIG["INTERVAL"]
                idx += 1
                if idx >= self._c_pos_otg_d.shape[1]:
                    idx = 0
                
                traj.append(
                    [CONFIG["INTERVAL"],] + list(self.tstate["cp"]) + list(self.tstate["tp"])
                )
                if time >= self.max_time:
                    idx, time, click_success = 0, 0, 0
                    initial_cond = self._sample_initial_condition(fixed_initial_cond)
                    traj = list([
                        [0,] + list(self.tstate["cp"]) + list(self.tstate["tp"]),
                    ])
                    continue
            else:
                time_offset = click_time - (idx * CONFIG["INTERVAL"])
                self._tstate_step(idx, time_offset)
                time += time_offset

                traj.append(
                    [time_offset,] + list(self.tstate["cp"]) + list(self.tstate["tp"])
                )
                click_dist = ((self.tstate["cp"] - self.tstate["tp"]) ** 2).sum() ** 0.5
                click_success = click_dist < self.tstate["tr"][0]
                done = True

        click_dist = np.clip(click_dist, 0, self.max_dist)
        stat = np.concatenate([
            np.array([click_success, time / self.max_time, click_dist / self.max_dist]),
            initial_cond
        ])
        return stat.astype(np.float32), np.array(traj, dtype=np.float32)

    def _sample_initial_condition(self, fixed_initial_cond=None):
        min_tv, max_tv = -0.36, 0.36
        min_tr, max_tr = 0.0096, 0.024
        if fixed_initial_cond is None:
            self.tstate = {
                "cp": self.simul_random.uniform(
                    low=[0.0, 0.0],
                    high=[CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_HEIGHT"]],
                    size=(2,),
                ),
                "cv": self.simul_random.normal(loc=0, scale=0.1, size=(2,)),
                "tp": self.simul_random.uniform(
                    low=[0.0, 0.0],
                    high=[CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_HEIGHT"]],
                    size=(2,),
                ),
                "tv": self.simul_random.uniform(low=min_tv, high=max_tv, size=(2,)),
                "tr": self.simul_random.uniform(low=min_tr, high=max_tr, size=(1,)),
                "hp": np.array([0.0, 0.0]),
            }
        else:
            self.tstate = {
                "cp": fixed_initial_cond[0:2] * np.array([CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_HEIGHT"]]),
                "cv": fixed_initial_cond[2:4],
                "tp": fixed_initial_cond[4:6] * np.array([CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_HEIGHT"]]),
                "tv": fixed_initial_cond[6:8] * max_tv,
                "tr": fixed_initial_cond[8:] * max_tr,
                "hp": np.array([0.0, 0.0]),
            }

        self.cstate = deepcopy(self.tstate)
        self.cstate["tp"], self.cstate["tv"] = visual.add_noise_past(
            int(CONFIG["Tp"] // CONFIG["INTERVAL"]),
            self.tstate["tp"],
            self.tstate["tv"],
            self.tstate["tr"],
            self.free_params["sigmav"]
        )

        initial_cond = np.concatenate([
            self.tstate["cp"] / np.array([CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_HEIGHT"]]),
            self.tstate["cv"],
            self.tstate["tp"] / np.array([CONFIG["WINDOW_WIDTH"], CONFIG["WINDOW_HEIGHT"]]),
            self.tstate["tv"] / max_tv,
            self.tstate["tr"] / max_tr,
        ])
        return initial_cond

    def _observation(self):
        return np.concatenate([
            self.cstate["cp"],
            self.cstate["cv"],
            self.cstate["tp"],
            self.cstate["tv"],
            self.cstate["tr"] * 10, # Multiplied by 10 when used as inputs to DQN
            self.cstate["hp"],
        ])

    def _get_action(self, obs):
        return np.argmax(self.qnet(obs, z=self._z).detach().cpu().numpy()) 

    def _set_bump(self, act):      
        (
            cursor_info,
            target_info,
            hand_info,
            click_info,
            effort_info
        ) = pnc_model.model(
            self.tstate, self.cstate, act, self.free_params
        )
        (
            _c_pos_otg_dx,
            _c_pos_otg_dy,
            _c_vel_otg_x,
            _c_vel_otg_y,
            c_pos_delta,
            c_vel_ideal,
        ) = cursor_info
        self._c_pos_otg_d = np.vstack([_c_pos_otg_dx, _c_pos_otg_dy])
        self._c_vel_otg = np.vstack([_c_vel_otg_x, _c_vel_otg_y])

        _h_pos_x, _h_pos_y, _h_pos_delta_x, _h_pos_delta_y = hand_info
        self._h_pos_ideal = np.array([_h_pos_x, _h_pos_y])
        h_pos_delta = np.array([_h_pos_delta_x, _h_pos_delta_y])

        # For next BUMP
        self.cstate["cp"] = self.tstate["cp"] + c_pos_delta
        self.cstate["cv"] = c_vel_ideal
        self.cstate["tp"], self.cstate["tv"] = motor.boundary(
            self._c_pos_otg_d.shape[1],
            target_info[0:2],
            target_info[2:4],
            self.tstate["tr"],
        )
        self.cstate["hp"] = self.tstate["hp"] + h_pos_delta
        return click_info

    def _tstate_step(self, idx, time_offset=CONFIG["INTERVAL"]):
        idx_offset = time_offset / CONFIG["INTERVAL"]
        self.tstate["cp"] += self._c_pos_otg_d[:, idx] * idx_offset
        self.tstate["cv"] = self._c_vel_otg[:, idx + 1]
        self.tstate["tp"], self.tstate["tv"] = motor.boundary(
            idx_offset,
            self.tstate["tp"],
            self.tstate["tv"],
            self.tstate["tr"],
        )
        # hand pos is not estimated for every interval
        self.tstate["hp"] = self._h_pos_ideal

    def seeding(self, seed):
        self.simul_random = np.random.default_rng(seed)
        self.param_random = np.random.default_rng(seed)