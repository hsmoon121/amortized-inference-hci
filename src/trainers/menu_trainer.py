from time import time
from copy import deepcopy
import numpy as np
import torch

from .base_trainer import BaseTrainer
from ..nets import AmortizerForSummaryData
from ..simulators import MenuSimulator
from ..datasets import MenuUserDataset, MenuTrainDataset, MenuValidDataset
from ..utils import ReplayBuffer, CosAnnealWR
from ..configs import default_menu_config


class MenuTrainer(BaseTrainer):
    def __init__(self, config=None):
        self.config = deepcopy(default_menu_config) if config is None else config
        super().__init__(name=self.config["name"], task_name="menu")

        # Initialize the amortizer, simulator, and datasets
        self.amortizer = AmortizerForSummaryData(config=self.config["amortizer"])
        self.simulator = MenuSimulator(config=self.config["simulator"])
        self.user_dataset = MenuUserDataset()
        self.valid_dataset = MenuValidDataset(n_param=200, sim_per_param=5000)

        self.targeted_params = self.config["simulator"]["targeted_params"]
        self.param_symbol = np.array(self.config["simulator"]["param_symbol"])
        self.base_params = np.array(self.config["simulator"]["base_params"])
        self.obs_label = ["pre_tct", "pre_nfix", "abs_tct", "abs_nfix"]
        self.obs_description = [
            "Completion time [ms] (w/ target)",
            "Num. of fixations (w/ target)",
            "Completion time [ms] (w/o target)",
            "Num. of fixations (w/o target)",
        ]

        # Initialize the optimizer and scheduler
        self.lr = self.config["learning_rate"]
        self.lr_gamma = self.config["lr_gamma"]
        self.clipping = self.config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

    def train(
        self,
        n_iter=300,
        step_per_iter=2000,
        batch_sz=512,
        n_trial=256,
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
            self.memory = ReplayBuffer(capacity, use_traj=False)
        elif train_mode == "online":
            pass
        elif train_mode == "offline":
            self.train_dataset = MenuTrainDataset(
                total_sim=2**26,
                n_ep=256,
                sim_config=self.config["simulator"],
            )
        else:
            raise RuntimeError(f"Wrong training type: {train_mode}")

    def _get_online_batch(self, batch_sz, n_trial):
        assert n_trial >= 100 # to avoid zeros in the summary statistics
        return self.simulator.simulate(n_param=batch_sz, sim_per_param=n_trial)

    def _get_offline_batch(self, batch_sz, n_trial):
        assert hasattr(self, "train_dataset")
        return self.train_dataset.sample(
            batch_sz=batch_sz,
        )

    def _get_replay_batch(self, batch_sz, n_trial):
        assert hasattr(self, "memory")
        assert n_trial >= 100 # to avoid zeros in the summary statistics

        if len(self.memory) < batch_sz * 2:
            sim_args = self.simulator.simulate(
                n_param=batch_sz * 2,
                sim_per_param=n_trial,
            )
        else:
            sim_args = self.simulator.simulate(
                n_param=1,
                sim_per_param=n_trial,
            )
        self.memory.push(*sim_args)
        return self.memory.sample(batch_sz)

    def valid(
        self,
        n_sample=10000,
        infer_type="mode",
        plot=True,
        verbose=True
    ):
        self.amortizer.eval()
        valid_res = dict()

        ### 1) Parameter recovery from simulated dataset
        start_t = time()

        gt_params, valid_data = self.valid_dataset.sample()
        self.parameter_recovery(
            valid_res,
            gt_params,
            valid_data,
            n_sample,
            infer_type,
            plot=plot,
        )
        if verbose:
            print(f"- parameter recovery ({time() - start_t:.3f}s)")

        # Get user data for fitting (group-level)
        group_data_for_fitting, indiv_data_for_validation = self.user_dataset.indiv_sample(
            cross_valid=True,
            for_pop=True,
        )
        group_data_for_fitting = np.expand_dims(group_data_for_fitting, axis=0)

        # Set user data for validation
        validation_data = dict(zip(self.obs_label, [list(), list(), list(), list()]))
        validation_n_trials = list()
        for ub in indiv_data_for_validation:
            for key in validation_data:
                validation_data[key] += ub[key]
            validation_n_trials.append(len(ub["pre_tct"]) + len(ub["abs_tct"]))

        if plot:
            self._plot_histogram(validation_data, "user_observations", color="#EE7733")

        ### 2) Group-level user dataset fitting
        start_t = time()

        group_level_params = self._group_level_fitting(
            valid_res,
            group_data_for_fitting,
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
            validation_data,
            plot=plot
        )
        if verbose:
            print(f"- group-level simulation with fitted params ({time() - start_t:.3f}s)")

        ### 4) Individual-level user dataset fitting
        start_t = time()
        
        indiv_data_for_fitting, _ = self.user_dataset.indiv_sample(cross_valid=True, for_pop=False)
        indiv_inferred_params = self._indiv_level_fitting(
            indiv_data_for_fitting,
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
            validation_data,
            validation_n_trials,
            plot=plot
        )
        if verbose:
            print(f"- individual-level simulation with fitted params ({time() - start_t:.3f}s)")

        return valid_res
    
    def _group_level_fitting(
        self,
        res,
        group_data,
        n_sample,
        infer_type,
        plot=False
    ):
        """
        Group-level fitting on user dataset
        """
        start = time()
        inferred_params, samples = self.amortizer.infer(
            group_data,
            n_sample=n_sample,
            type=infer_type,
            return_samples=True
        )
        inferred_params = self._clip_params(inferred_params)
        infer_time = time() - start

        for i, l in enumerate(self.targeted_params):
            res["Inferred_Params/user_" + l] = inferred_params[i]
        res["Inference_Time/infer_time"] = infer_time

        if plot:
            self._pair_plot(
                samples,
                self.param_symbol,
                limits=[[0., 6.], [0., 1.], [0., 1.], [0., 1.]],
                gt_params=self.base_params,
                fname="user_posterior"
            )
        return inferred_params
    
    def _group_level_prediction(
        self,
        res,
        group_level_params,
        validation_data,
        plot=False
    ):
        """
        Group-level & baseline params' prediction on validation dataset
        """
        # Simulation with baseline parameters (CHI'17 results)
        _, _, prev_sim_behavior = self.simulator.simulate(
            sim_per_param=10000,
            fixed_params=self.base_params,
            verbose=False,
            return_behavior=True,
        )
        # Simulation with group-level inferred parameters
        _, _, group_sim_behavior = self.simulator.simulate(
            sim_per_param=10000,
            fixed_params=group_level_params,
            verbose=False,
            return_behavior=True,
        )
        if plot:
            self._plot_histogram(prev_sim_behavior, "base_sim_observations", color="#0077BB")
            self._plot_histogram(group_sim_behavior, "group_sim_observations", color="#009A3E")

        for l, b in zip(["prev", "group"], [prev_sim_behavior, group_sim_behavior]):
            keys, vals = self.behavior_distance(validation_data, b, metrics=self.obs_label, label=l)
            for k, v in zip(keys, vals):
                res[k] = v

    def _indiv_level_fitting(
        self,
        indiv_data,
        n_sample,
        infer_type
    ):
        """
        Individual-level fitting on user dataset
        """
        indiv_inferred_params = list()
        for user_i in range(len(indiv_data)):
            # Inference with individual-level data
            inferred_params = self.amortizer.infer(
                np.expand_dims(indiv_data[user_i], axis=0),
                n_sample=n_sample,
                type=infer_type,
                return_samples=False
            )
            inferred_params = self._clip_params(inferred_params)
            indiv_inferred_params.append(inferred_params)

        return indiv_inferred_params
    
    def _indiv_level_prediction(
        self,
        res,
        indiv_inferred_params,
        validation_data,
        validation_n_trials,
        plot=False
    ):
        """
        Individual-level prediction on validation dataset
        """
        # Simulation with individual-level inferred parameters
        indiv_sim_behavior = dict(zip(self.obs_label, [list(), list(), list(), list()]))
        for user_i in range(len(indiv_inferred_params)):
            _, _, b = self.simulator.simulate(
                sim_per_param=validation_n_trials[user_i] * 10,
                fixed_params=indiv_inferred_params[user_i],
                verbose=False,
                return_behavior=True,
            )
            for key in indiv_sim_behavior:
                indiv_sim_behavior[key] += b[key]

        if plot:
            self._plot_histogram(indiv_sim_behavior, "indiv_sim_observations", color="#009A3E")

        keys, vals = self.behavior_distance(validation_data, indiv_sim_behavior, metrics=self.obs_label, label="indiv")
        for k, v in zip(keys, vals):
            res[k] = v
    
    def _behavior_bin_info(self, label):
        if label.endswith("tct"):
            minbin = 0.0
            maxbin = 3000.0
            nbins = 8
        elif label.endswith("nfix"):
            minbin = 0.0
            maxbin = 10.0
            nbins = 11
        else:
            raise RuntimeError("Unknown obs label: {}".format(label))
        return minbin, maxbin, nbins

    def _clip_params(self, params):
        return np.clip(
            params,
            np.array([0, 0, 0., 0.]),
            np.array([np.inf, np.inf, 1., 1.])
        )