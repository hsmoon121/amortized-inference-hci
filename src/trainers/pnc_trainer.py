from time import time
from copy import deepcopy
import numpy as np
import torch

from .base_trainer import BaseTrainer
from ..nets import AmortizerForTrialData
from ..simulators import PnCSimulator
from ..datasets import PnCUserDataset, PnCTrainDataset, PnCValidDataset
from ..utils import ReplayBuffer, CosAnnealWR
from ..configs import default_pnc_config


class PnCTrainer(BaseTrainer):
    def __init__(self, config=None):
        self.config = deepcopy(default_pnc_config) if config is None else config
        super().__init__(name=self.config["name"], task_name="pnc")

        # Initialize the amortizer, simulator, and datasets
        self.amortizer = AmortizerForTrialData(config=self.config["amortizer"])
        self.simulator = PnCSimulator(config=self.config["simulator"])
        self.user_dataset = PnCUserDataset()
        self.valid_dataset = PnCValidDataset(sim_config=self.config["simulator"])

        self.targeted_params = self.config["simulator"]["targeted_params"]
        self.param_symbol = np.array(self.config["simulator"]["param_symbol"])
        self.base_params = np.array(self.config["simulator"]["base_params"])
        self.obs_label = ["tct", "click_dist_norm", "travel_dist"]
        self.obs_description = [
            "Completion time [s]",
            "Click endpoint (normalized)",
            "Cursor travel distance [m]",
        ]

        # Initialize the optimizer and scheduler
        self.lr = self.config["learning_rate"]
        self.lr_gamma = self.config["lr_gamma"]
        self.clipping = self.config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

    def train(
        self,
        n_iter=500,
        step_per_iter=2000,
        batch_sz=32,
        n_trial=32,
        train_mode="replay",
        capacity=10000,
        board=True
    ):
        super().train(
            n_iter=n_iter,
            step_per_iter=step_per_iter,
            batch_sz=batch_sz,
            n_trial=n_trial,
            train_mode=train_mode,
            capacity=capacity,
            board=board,
        )

    def _set_train_mode(self, train_mode, capacity):
        """
        Set the training mode for the trainer ("replay", "online", "offline")
        """
        if train_mode == "replay":
            self.memory = ReplayBuffer(capacity, use_traj=True)
        elif train_mode == "online":
            pass
        elif train_mode == "offline":
            total_sim = 2**23 if "max_th" in self.targeted_params else 2**22
            self.train_dataset = PnCTrainDataset(
                total_sim=total_sim,
                sim_config=self.config["simulator"]
            )
        else:
            raise RuntimeError(f"Wrong training type: {train_mode}")

    def _get_online_batch(self, batch_sz, n_trial):
        return self.simulator.simulate(n_param=batch_sz, sim_per_param=n_trial)

    def _get_offline_batch(self, batch_sz, n_trial):
        assert hasattr(self, "train_dataset")
        return self.train_dataset.sample(batch_sz=batch_sz, sim_per_param=n_trial)

    def _get_replay_batch(self, batch_sz, n_trial):
        assert hasattr(self, "memory")
        if len(self.memory) < batch_sz * 2:
            sim_args = self.simulator.simulate(n_param=batch_sz * 2, sim_per_param=n_trial)
        else:
            sim_args = self.simulator.simulate(n_param=1, sim_per_param=n_trial)
        self.memory.push(*sim_args)
        return self.memory.sample(batch_sz)

    def valid(
        self,
        n_trial=200,
        n_sample=100,
        infer_type="mode",
        plot=True,
        verbose=True
    ):
        self.amortizer.eval()
        valid_res = dict()

        ### 1) Parameter recovery from simulated & user dataset
        start_t = time()

        sim_gt_params, sim_valid_data = self.valid_dataset.sample(n_trial)
        user_gt_params = self.user_dataset.gt_data
        if "max_th" in self.targeted_params:
            user_gt_params = np.concatenate((
                user_gt_params,
                np.ones((user_gt_params.shape[0], 1))
            ), axis=-1)
        user_valid_data = self.user_dataset.sample(n_trial)

        for gt_params, valid_data, surfix in zip(
            [sim_gt_params, user_gt_params],
            [sim_valid_data, user_valid_data],
            ["_sim", "_user"]
        ):
            self.parameter_recovery(
                valid_res,
                gt_params,
                valid_data,
                n_sample,
                infer_type,
                plot=plot,
                surfix=surfix,
            )
        if verbose:
            print(f"- parameter recovery (simulated & user dataset) ({time() - start_t:.3f}s)")

        # Get user data for fitting (group-level)
        data_for_fitting, data_for_validation = self.user_dataset.sample(
            n_trial,
            indiv_user=True,
            cross_valid=True,
        )

        ### 2) Group-level user dataset fitting
        start_t = time()

        group_level_params = self._group_level_fitting(
            valid_res,
            data_for_fitting,
            n_sample,
            infer_type,
            plot=plot,
        )
        if verbose:
            print(f"- group-level inference from user dataset ({time() - start_t:.3f}s)")

        ### 3) Group-level prediction on user dataset
        start_t = time()
        
        self._group_level_prediction(
            valid_res,
            group_level_params,
            data_for_validation,
            plot=plot
        )
        if verbose:
            print(f"- group-level simulation with fitted params ({time() - start_t:.3f}s)")

        ### 4) Individual-level user dataset fitting
        start_t = time()
        
        indiv_inferred_params = self._indiv_level_fitting(
            data_for_fitting,
            n_sample,
            infer_type
        )
        if verbose:
            print(f"- individual-level inference from user dataset ({time() - start_t:.3f}s)")

        ### 5) Individual-level prediction prediction on user dataset
        start_t = time()
        
        self._indiv_level_prediction(
            valid_res,
            indiv_inferred_params,
            data_for_validation,
            plot=plot
        )
        if verbose:
            print(f"- individual-level simulation with fitted params ({time() - start_t:.3f}s)")

        return valid_res
    
    def _group_level_fitting(
        self,
        res,
        data_for_fitting,
        n_sample,
        infer_type,
        plot=False
    ):
        """
        Group-level fitting on user dataset
        """
        group_stat_data = np.concatenate([ud[0] for ud in data_for_fitting], axis=0)
        group_traj_data = list()
        for ud in data_for_fitting:
            group_traj_data += ud[1]

        start = time()
        lognorm_params, lognorm_samples = self.amortizer.infer(
            group_stat_data,
            group_traj_data,
            n_sample=n_sample,
            type=infer_type,
            return_samples=True
        )
        lognorm_params = self._clip_params(lognorm_params)
        inferred_params = self.simulator.convert_from_output(lognorm_params)[0]
        samples = self.simulator.convert_from_output(lognorm_samples)
        infer_time = time() - start

        for i, l in enumerate(self.targeted_params):
            res["Inferred_Params/user_" + l] = inferred_params[i]
        res["Inference_Time/infer_time"] = infer_time

        if plot:
            # Change order [n_v, sigma_v, ...] --> [sigma_v, n_v, ...]
            permutation = np.arange(len(self.targeted_params))
            permutation[0], permutation[1] = 1, 0
            self._pair_plot(
                samples[:, permutation],
                self.param_symbol[permutation],
                gt_params=self.base_params[permutation],
                fname="user_posterior"
            )
        return inferred_params
    
    def _group_level_prediction(
        self,
        res,
        group_level_params,
        data_for_validation,
        plot=False
    ):
        """
        Group-level & baseline params' prediction on validation dataset
        """
        # Simulation with baseline parameters (measured user capability values in CHI'22)
        gt_params = self.user_dataset.gt_data
        prev_sim_data = list()
        for _, gt_param in enumerate(gt_params):
            _, prev_sim_stats, prev_sim_trajs = self.simulator.simulate(
                fixed_params=gt_param,
                sim_per_param=10000 // gt_params.shape[0],
                verbose=False,
            )
            prev_sim_data.append([prev_sim_stats[0], prev_sim_trajs[0]])

        # Simulation with group-level inferred parameters
        _, sim_stats, sim_trajs = self.simulator.simulate(
            fixed_params=group_level_params,
            sim_per_param=10000,
            verbose=False,
        )

        user_behavior = self.get_behavior(data_for_validation)
        prev_sim_behavior = self.get_behavior(prev_sim_data)
        group_sim_behavior = self.get_behavior([[sim_stats[0], sim_trajs[0]]])

        if plot:
            self._plot_histogram(user_behavior, "user_observations", color="#EE7733")
            self._plot_histogram(prev_sim_behavior, "base_sim_observations", color="#0077BB")
            self._plot_histogram(group_sim_behavior, "group_sim_observations", color="#009A3E")

        for l, b in zip(["prev", "group"], [prev_sim_behavior, group_sim_behavior]):
            keys, vals = self.behavior_distance(user_behavior, b, metrics=self.obs_label, label=l)
            for k, v in zip(keys, vals):
                res[k] = v

    def _indiv_level_fitting(
        self,
        data_for_fitting,
        n_sample,
        infer_type
    ):
        """
        Individual-level fitting on user dataset
        """
        indiv_inferred_params = list()
        for user_i, indiv_ud in enumerate(data_for_fitting):
            # Inference with individual-level data
            lognorm_params = self.amortizer.infer(
                indiv_ud[0],
                indiv_ud[1],
                n_sample=n_sample,
                type=infer_type,
                return_samples=False
            )
            lognorm_params = self._clip_params(lognorm_params)
            inferred_params = self.simulator.convert_from_output(lognorm_params)[0]
            indiv_inferred_params.append(inferred_params)

        return indiv_inferred_params
    
    def _indiv_level_prediction(
        self,
        res,
        indiv_inferred_params,
        data_for_validation,
        plot=False
    ):
        """
        Individual-level prediction on validation dataset
        """
        # Simulation with individual-level inferred parameters
        indiv_sim_data = list()
        for user_i, indiv_ud in enumerate(data_for_validation):
            indiv_n_trial = len(indiv_ud[1]) * 5
            _, indiv_sim_stats, indiv_sim_trajs = self.simulator.simulate(
                fixed_params=indiv_inferred_params[user_i],
                sim_per_param=indiv_n_trial,
                verbose=False,
            )
            indiv_sim_data.append([indiv_sim_stats[0], indiv_sim_trajs[0]])

        indiv_sim_behavior = self.get_behavior(indiv_sim_data)
        if plot:
            self._plot_histogram(indiv_sim_behavior, "indiv_sim_observations", color="#009A3E")

        user_behavior = self.get_behavior(data_for_validation)
        keys, vals = self.behavior_distance(user_behavior, indiv_sim_behavior, metrics=self.obs_label, label="indiv")
        for k, v in zip(keys, vals):
            res[k] = v

    def get_behavior(self, user_data):
        max_time = 5.0
        max_dist = 5 * 0.024
        max_radius = 0.024
        behavior = dict()
        stat_list, traj_list = list(), list()
        for user in user_data:
            stat_list.append(user[0])
            traj_list += user[1]
        stats = np.concatenate(stat_list, axis=0)

        mean_vel_list = list()
        travel_dist_list = list()
        for traj in traj_list:
            dist = np.linalg.norm(traj[1:, 1:3] - traj[:-1, 1:3], axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                vel = dist / traj[1:, 0]
            mean_vel_list.append(np.nanmean(vel))
            travel_dist_list.append(np.nansum(dist))
        mean_vel = np.array(mean_vel_list)
        travel_dist = np.array(travel_dist_list)

        behavior["tct"] = stats[:, 1] * max_time
        behavior["click_dist_norm"] = (stats[:, 2] * max_dist) / (stats[:, -1] * max_radius)
        behavior["travel_dist"] = travel_dist[:]
        return behavior
    
    def _behavior_bin_info(self, label):
        if label.startswith("tct"):
            minbin = 0.
            maxbin = 2.
            nbins = 10
        elif label.startswith("click_dist_norm"):
            minbin = 0.
            maxbin = 4.0
            nbins = 12
        elif label.startswith("travel_dist"):
            minbin = 0.
            maxbin = 0.6
            nbins = 12
        else:
            raise RuntimeError("Unknown obs label: {}".format(label))
        return minbin, maxbin, nbins

    def _clip_params(self, params):
        return np.clip(
            params,
            np.array([-1.] * len(self.targeted_params)),
            np.array([1.] * len(self.targeted_params))
        )