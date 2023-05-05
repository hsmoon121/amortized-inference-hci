import os, pickle
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
from joblib import Parallel, delayed
from datetime import datetime

from ..simulators import PnCSimulator
from ..simulators.pnc.model_config import CONFIG, STD_PARAMS, MEAN_PARAMS
from ..configs import default_pnc_config


class PnCUserDataset(object):
    """
    The class 'PnCUserDataset' handles the creation and retrieval of an empirical dataset for Point-and-Click (PnC) tasks.
    ==> From "Speeding up Inference with User Simulators through Policy Modulation" (CHI 2022) by Moon et al.
    ==> https://github.com/hsmoon121/pnc-dataset
    """
    def __init__(self):
        self.n_user = 20
        self.max_time = 5.0
        self.max_tv = 0.36
        self.max_tr = 0.024
        self.max_dist = 5 * self.max_tr
        self._get_stat_data()
        self._get_traj_data()
        self._get_gt_data()

    def _get_stat_data(self):
        """
        Read static (fixed-size) behavioral outputs for every trial from CSV file
        It removes the first block and first trial of each block, normalizes the data and identifies outlier trials.
        """
        df_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "data/pnc/datasets/empirical_data_stat.csv"
        )
        exp_df = pd.read_csv(df_path)
        self.stat_data = list()
        self.outlier_data = list()
        for user in range(self.n_user):
            # remove 1st block & 1st trial of each block
            user_df = exp_df[
                (exp_df["user"] == user) \
                & (exp_df["task"] > 0) \
                & (exp_df["trial"] > 0)
            ]
            max_values = np.array([
                1,
                self.max_time,
                CONFIG["WINDOW_WIDTH"],
                CONFIG["WINDOW_HEIGHT"],
                1,
                1,
                CONFIG["WINDOW_WIDTH"],
                CONFIG["WINDOW_HEIGHT"],
                self.max_tv,
                self.max_tv,
                self.max_tr
            ])
            user_data = user_df[[
                "success", "time", "c_pos_x", "c_pos_y", "c_vel_x", "c_vel_y",
                "t_pos_x", "t_pos_y", "t_vel_x", "t_vel_y", "radius"
            ]].to_numpy(copy=True) / max_values

            outlier_idx = np.where((user_data[:, 1] > 1.0) | (user_data[:, 1] < 0.05))[0]
            mask = np.ones((user_data.shape[0],), bool)
            mask[outlier_idx] = False
            self.outlier_data.append(outlier_idx)
            self.stat_data.append(user_data[mask, :])

    def _get_traj_data(self):
        """
        Read trajectory (variable-size) behavioral outputs for every trial from pickle file
        """
        df_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "data/pnc/datasets/empirical_data_traj.pkl"
        )
        traj_df = pd.read_pickle(df_path)
        self.traj_data = list()
        for user in range(self.n_user):
            user_traj = []
            len_traj = []
            click_dist = []
            # remove 1st block & 1st trial of each block
            for bl in range(1, 4):
                for tr in range(1, 200):
                    idx = 199 * (bl - 1) + (tr - 1)
                    if idx not in self.outlier_data[user]:
                        traj = traj_df[user][bl][tr]
                        traj[1:, 0] = traj[1:, 0] - traj[:-1, 0] # relative time
                        user_traj.append(traj)
                        len_traj.append(len(traj))

                        final_d = ((traj[-1, 1:3] - traj[-1, 3:5]) ** 2).sum() ** 0.5
                        click_dist.append(np.clip(final_d, 0, self.max_dist))
                        
            # append click distance to stat_data
            self.stat_data[user] = np.insert(
                self.stat_data[user],
                2,
                np.array(click_dist) / self.max_dist,
                axis=-1
            )
            self.traj_data.append(user_traj)

    def _get_gt_data(self, lognorm=False, set_outlier=True):
        """
        Read the ground-truth model parameter values from a CSV file.
        It processes them, optionally setting the outlier values (P13 & P14) and applying log-normal scaling.
        """
        df_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "data/pnc/datasets/empirical_data_param.csv"
        )
        gt_param = pd.read_csv(df_path)
        max_sigmav = MEAN_PARAMS["sigmav"] + 3 * STD_PARAMS["sigmav"]
        if set_outlier:
            gt_param["sigmav"][13] = max_sigmav
            gt_param["sigmav"][14] = max_sigmav

        param_label = ["nv", "sigmav", "csigma"]
        if lognorm:
            lognorm_data = list()
            for l in param_label:
                scale = gt_param[l].to_numpy() / MEAN_PARAMS[l]
                scale_base = 1 + 3 * STD_PARAMS[l] / MEAN_PARAMS[l]
                lognorm_data.append(np.log(scale) / np.log(scale_base))
            self.gt_data = np.array(lognorm_data).transpose()
        else:        
            self.gt_data = np.vstack([
                gt_param[l].to_numpy(copy=True) for l in param_label
            ]).transpose()

    def _sample_user_trial(self, user, n_trial):
        """
        Sample the specified number of trials for a given user.
        If the requested number of trials is larger than the available data,
        it repeats the data until the required number of trials is reached.
        """
        if n_trial > self.stat_data[user].shape[0]:
            stats = self.stat_data[user]
            trajs = list()
            trajs += self.traj_data[user]
            diff = n_trial - self.stat_data[user].shape[0]
            while diff > self.stat_data[user].shape[0]:
                stats = np.concatenate((stats, self.stat_data[user]), axis=0)
                trajs += self.traj_data[user]
                diff -= self.stat_data[user].shape[0]
            stats = np.concatenate((stats, self.stat_data[user][-diff:]), axis=0)
            trajs += self.traj_data[user][-diff:]
        else:
            stats = self.stat_data[user][-n_trial:]
            trajs = self.traj_data[user][-n_trial:]
        return stats, trajs

    def sample(self, n_trial, n_user=20, indiv_user=False, cross_valid=False):
        """
        Sample data from the dataset with a specified number of trials and users.
        It supports individual user sampling (indiv_user=True) and cross-validation settings (cross_valid=True). 
        - i) If both flags are set, the method splits the data into training and validation sets for each user.
        - ii) If only cross_valid is set, it splits the data into two groups,
              with half of the users used for training and the other half for validation.
        - iii) If neither flag is set, it returns the entire dataset.
        """
        user_data, valid_user_data = list(), list()
        for user in range(n_user):
            stats, trajs = self._sample_user_trial(user, n_trial)

            if indiv_user:
                if cross_valid:
                    half = n_trial // 2
                    user_data.append([stats[:half], trajs[:half]])
                    valid_user_data.append([stats[half:], trajs[half:]])
                else:
                    user_data.append([stats, trajs])
            else:
                if user >= n_user // 2 and cross_valid:
                    valid_user_data.append([stats, trajs])
                else:
                    user_data.append([stats, trajs])

        if cross_valid:
            return user_data, valid_user_data
        else:
            return user_data


class PnCTrainDataset(object):
    """
    The class 'PnCTrainDataset' handles the creation and retrieval of a training dataset with point-and-click simulator.
    It allows you to sample data from the dataset and supports various configurations.
    """
    def __init__(self, total_sim=2**23, n_ep=32, sim_config=None):
        """
        Initialize the dataset object with a specified number of total simulations, episodes,
        and a simulation configuration.
        """
        if sim_config is None:
            self.sim_config = deepcopy(default_pnc_config["simulator"])
        else:
            self.sim_config = sim_config
        self.sim_config["seed"] = 100
        self.total_sim = total_sim
        self.n_ep = n_ep
        self.n_param = total_sim // n_ep

        self.name = f"{self.total_sim//1000000}M_step_{self.n_ep}ep"
        if "max_th" not in self.sim_config["targeted_params"]:
            self.name += f"_wo-th"
        if self.sim_config["prior"] == "uniform" or self.sim_config["prior"] == "log-uniform":
            self.name += f"_{self.sim_config['prior']}"

        self.fpath = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/pnc/datasets/train_{self.name}.pkl"
        )
        self._get_dataset()

    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the PnCSimulator.
        """
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            print(f"[ train dataset ] {self.fpath}")
            self.simulator = PnCSimulator(self.sim_config)

            def get_simul_res(simulator, i):
                simulator.seeding(datetime.now().microsecond + i)
                args = simulator.simulate(
                    n_param=1,
                    sim_per_param=self.n_ep,
                    verbose=False
                )
                return args

            # Parallelize the creation of the dataset.
            num_cpus = psutil.cpu_count(logical=False)
            eps = Parallel(n_jobs=num_cpus - 1)(
                delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.n_param))
            )
            params_arr = np.concatenate([eps[i][0] for i in range(self.n_param)], axis=0, dtype=np.float32)
            stats_arr = np.concatenate([eps[i][1] for i in range(self.n_param)], axis=0, dtype=np.float32)
            trajs_arr = np.concatenate([eps[i][2] for i in range(self.n_param)], axis=0, dtype=object)

            self.dataset = dict(
                params=params_arr,      # np.array (n_param, param_sz)
                stat_data=stats_arr,    # np.array (n_param, n_ep, stat_sz)
                traj_data=trajs_arr,    # np.array (n_param, n_ep) of (T, traj_sz)
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f)

    def sample(self, batch_sz, sim_per_param=1):
        """
        Returns a random sample from the dataset with the specified number of parameter sets (batch size),
        and number of simulated trials per parameters (sim_per_param).
        """
        indices = np.random.choice(self.n_param, batch_sz)
        ep_indices = np.random.choice(self.n_ep, sim_per_param)
        rows = np.repeat(indices, sim_per_param).reshape((-1, sim_per_param))
        cols = np.tile(ep_indices, (batch_sz, 1))
        if sim_per_param == 1:
            return (
                self.dataset["params"][indices],
                self.dataset["stat_data"][rows, cols].squeeze(1),
                self.dataset["traj_data"][rows, cols].squeeze(1),
            )
        else:
            return (
                self.dataset["params"][indices],
                self.dataset["stat_data"][rows, cols],
                self.dataset["traj_data"][rows, cols],
            )


class PnCValidDataset(object):
    """
    The class 'PnCValidDataset' handles the creation and retrieval of a validation dataset with point-and-click simulator.
    """
    def __init__(self, total_user=100, trial_per_user=600, sim_config=None):
        """
        Initialize the dataset object with a specified number of total user (different parameter sets), episodes,
        and a simulation configuration.
        """
        self.total_user = total_user
        self.trial_per_user = trial_per_user
        if sim_config is None:
            self.sim_config = deepcopy(default_pnc_config["simulator"])
        else:
            self.sim_config = deepcopy(sim_config)
        self.sim_config["seed"] = 121
        
        self.name = f"{total_user}_param_{trial_per_user}ep"
        if "max_th" not in self.sim_config["targeted_params"]:
            self.name += f"_wo-th"
        if self.sim_config["prior"] == "uniform" or self.sim_config["prior"] == "log-uniform":
            self.name += f"_{self.sim_config['prior']}"

        self.fpath = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/pnc/datasets/valid_{self.name}.pkl"
        )
        self._get_dataset()

    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the PnCSimulator.
        """
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            print(f"[ valid dataset ] {self.fpath}")
            self.simulator = PnCSimulator(self.sim_config)

            def get_simul_res(simulator, i):
                simulator.seeding(datetime.now().microsecond + i)
                args = simulator.simulate(
                    n_param=1,
                    sim_per_param=self.trial_per_user,
                    verbose=False
                )
                return args

            num_cpus = psutil.cpu_count(logical=False)
            eps = Parallel(n_jobs=num_cpus - 1)(
                delayed(get_simul_res)(self.simulator, i) for i in tqdm(range(self.total_user))
            )
            params_arr = np.concatenate([
                self.simulator.convert_from_output(eps[i][0]) for i in range(self.total_user)
            ], axis=0, dtype=np.float32)
            stats_arr = np.concatenate([eps[i][1] for i in range(self.total_user)], axis=0, dtype=np.float32)
            trajs_arr = np.concatenate([eps[i][2] for i in range(self.total_user)], axis=0, dtype=object)

            self.dataset = dict(
                params=params_arr,      # np.array (n_param, param_sz)
                stat_data=stats_arr,    # np.array (n_param, n_ep, stat_sz)
                traj_data=trajs_arr,    # np.array (n_param, n_ep) of (T, traj_sz)
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f)

    def sample(self, n_trial, n_user=None, indiv_user=False, cross_valid=False):
        """
        Sample the dataset for the given number of trials and users.
        It returns the sampled data based on the specified options (indiv_user and cross_valid).
        If indiv_user is set to True, the data for each user is returned individually.
        If cross_valid is set to True, the data is divided into two parts for cross-validation purposes.
        """
        if n_user is None:
            n_user = self.total_user
        user_data, valid_user_data = list(), list()
        params = list()
        for user in range(n_user):
            params.append(self.dataset["params"][user])
            stats = self.dataset["stat_data"][user][:n_trial]
            trajs = self.dataset["traj_data"][user][:n_trial]

            if indiv_user:
                if cross_valid:
                    half = n_trial // 2
                    user_data.append([stats[:half], trajs[:half]])
                    valid_user_data.append([stats[half:], trajs[half:]])
                else:
                    user_data.append([stats, trajs])
            else:
                if user >= n_user // 2 and cross_valid:
                    valid_user_data.append([stats, trajs])
                else:
                    user_data.append([stats, trajs])

        if cross_valid:
            return np.array(params, dtype=np.float32), user_data, valid_user_data
        else:
            return np.array(params, dtype=np.float32), user_data