import os
from copy import deepcopy
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
import torch

from .base_trainer import BaseTrainer
from ..nets import AmortizerForTrialData, RegressionForTrialData
from ..simulators import TypingSimulator
from ..datasets import TypingUserDataset, TypingTrainDataset, TypingValidDataset
from ..utils import ReplayBuffer, CosAnnealWR
from ..configs import default_typing_config


class TypingTrainer(BaseTrainer):
    def __init__(self, config=None):
        self.config = deepcopy(default_typing_config) if config is None else config
        super().__init__(name=self.config["name"], task_name="typing")

        self.point_estimation = self.config["point_estimation"]
        amortizer_fn = RegressionForTrialData if self.point_estimation else AmortizerForTrialData

        # Initialize the amortizer, simulator, and datasets
        self.amortizer = amortizer_fn(config=self.config["amortizer"])
        self.simulator = TypingSimulator(config=self.config["simulator"])
        self.user_dataset = TypingUserDataset()
        self.valid_dataset = TypingValidDataset()

        self.targeted_params = self.config["simulator"]["targeted_params"]
        self.param_symbol = np.array(self.config["simulator"]["param_symbol"])
        self.base_params = np.array(self.config["simulator"]["base_params"])
        self.obs_label = ["wpm", "err_rate", "n_backspace", "kspc"]
        self.obs_description = ["WPM", "Error rate", "Num. of Backspaces", "KSPC"]

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
        n_trial=100,
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
            self.train_dataset = TypingTrainDataset(
                total_sim=2**20,
                n_ep=16,
                sim_config=self.config["simulator"],
            )
        else:
            raise RuntimeError(f"Wrong training type: {train_mode}")

    def _get_online_batch(self, batch_sz, n_trial):
        return self.simulator.simulate(n_param=batch_sz, n_eval_sentence=n_trial)

    def _get_offline_batch(self, batch_sz, n_trial):
        assert hasattr(self, "train_dataset")
        return self.train_dataset.sample(
            batch_sz=batch_sz,
            sim_per_param=n_trial,
        )
    
    def _get_replay_batch(self, batch_sz, n_trial):
        assert hasattr(self, "memory")
        if len(self.memory) < batch_sz * 2:
            sim_args = self.simulator.simulate(
                n_param=batch_sz * 2,
                n_eval_sentence=n_trial,
                random_sample=True,
            )
        else:
            sim_args = self.simulator.simulate(
                n_param=1,
                n_eval_sentence=n_trial,
                random_sample=True,
            )
        self.memory.push(*sim_args)
        return self.memory.sample(batch_sz)

    def valid(
        self,
        n_trial=100,
        n_sample=10000,
        infer_type="mode",
        plot=True,
        verbose=True
    ):
        self.amortizer.eval()
        valid_res = dict()

        ### 1) Parameter recovery from simulated dataset
        start_t = time()
        
        gt_params, valid_data = self.valid_dataset.sample(n_trial)
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
        group_data_for_fitting, data_for_validation, _ = self.user_dataset.indiv_sample(
            cross_valid=True,
            for_pop=True,
        )

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
            data_for_validation,
            plot=plot
        )
        if verbose:
            print(f"- group-level simulation with fitted params ({time() - start_t:.3f}s)")

        # Get user data for fitting (individual-level)
        indiv_data_for_fitting, _, indiv_user_info = self.user_dataset.indiv_sample(
            cross_valid=True,
            for_pop=False,
        )

        ### 4) Individual-level user dataset fitting
        start_t = time()
        
        indiv_inferred_params = self._indiv_level_fitting(
            indiv_data_for_fitting,
            indiv_user_info,
            n_sample,
            infer_type,
            plot=plot
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
        start = time()

        if self.point_estimation:
            inferred_params = self.amortizer.infer(data_for_fitting)
        else:
            inferred_params, samples = self.amortizer.infer(
                data_for_fitting,
                n_sample=n_sample,
                type=infer_type,
                return_samples=True
            )

            if plot:
                self._pair_plot(
                    samples,
                    self.param_symbol,
                    limits=[[0.0, 1.0], [0.1, 0.9], [0.04, 0.14]],
                    gt_params=self.base_params,
                    fname="user_posterior"
                )

        inferred_params = self._clip_params(inferred_params)
        infer_time = time() - start

        for i, l in enumerate(self.targeted_params):
            res["Inferred_Params/user_" + l] = inferred_params[i]
        res["Inference_Time/infer_time"] = infer_time

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
        # simulation with previous parameters
        _, prev_sim_stats = self.simulator.simulate(
            fixed_params=self.base_params,
            n_eval_sentence=2000,
            random_sample=False,
            verbose=False,
        )

        # simulation with population-level parameters
        _, sim_stats = self.simulator.simulate(
            fixed_params=group_level_params,
            n_eval_sentence=2000,
            random_sample=False,
            verbose=False,
        )

        user_behavior = self.get_behavior(data_for_validation)
        prev_sim_behavior = self.get_behavior(prev_sim_stats)
        group_sim_behavior = self.get_behavior(sim_stats)

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
        indiv_data_for_fitting,
        indiv_user_info,
        n_sample,
        infer_type,
        plot=False
    ):
        """
        Individual-level fitting on user dataset
        """
        cols = ["p_id", "age", "gender"] + self.targeted_params
        indiv_user_df = pd.DataFrame(columns=cols).astype(
            dict(zip(cols, [int, int, str, float, float, float]))
        )
        indiv_inferred_params = list()
        for user_i in range(len(indiv_data_for_fitting)):

            if self.point_estimation:
                inferred_params = self.amortizer.infer(indiv_data_for_fitting[user_i])
            else:
                inferred_params = self.amortizer.infer(
                    indiv_data_for_fitting[user_i],
                    n_sample=n_sample,
                    type=infer_type,
                    return_samples=False
                )
            inferred_params = self._clip_params(inferred_params)
            indiv_user_df = indiv_user_df.append(dict(zip(
                cols,
                [
                    indiv_user_info[user_i]["id"],
                    indiv_user_info[user_i]["age"],
                    indiv_user_info[user_i]["gender"],
                    inferred_params[0],
                    inferred_params[1],
                    inferred_params[2],
                ]
            )), ignore_index=True)
            indiv_inferred_params.append(inferred_params)

        if plot:
            self._plot_population(indiv_user_df)

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
        _, indiv_sim_stats = self.simulator.simulate(
            fixed_params=np.array(indiv_inferred_params),
            n_eval_sentence=4,
            random_sample=True,
            verbose=False,
        )
        indiv_sim_stats = indiv_sim_stats.reshape((-1, 5))
        indiv_sim_behavior = self.get_behavior(indiv_sim_stats)
        user_behavior = self.get_behavior(data_for_validation)

        if plot:
            self._plot_histogram(indiv_sim_behavior, "indiv_sim_observations", color="#009A3E")
        
        keys, vals = self.behavior_distance(user_behavior, indiv_sim_behavior, metrics=self.obs_label, label="indiv")
        for k, v in zip(keys, vals):
            res[k] = v

    def get_behavior(self, user_data):
        # Denormalize user data
        obs_max = np.array([80, 20, 20, 2.0, 80])
        obs_min = np.array([0, 0, 0, 0.5, 0])
        if len(user_data.shape) > 2:
            assert user_data.shape[0] == 1
            user_data = user_data.squeeze(axis=0)
        stats = (user_data + 1) / 2 * (obs_max - obs_min) + obs_min

        behavior = dict()
        for i, k in enumerate(self.obs_label):
            behavior[k] = stats[:, i]
        return behavior
    
    def _behavior_bin_info(self, label):
        if label == "wpm":
            minbin, maxbin = 0., 80.
        elif label == "err_rate":
            minbin, maxbin = 0., 20.
        elif label == "n_backspace":
            minbin, maxbin = 0., 20.
        elif label == "kspc":
            minbin, maxbin = 0.5, 2.0
        else:
            raise RuntimeError("Unknown obs label: {}".format(label))
        nbins = 10
        return minbin, maxbin, nbins

    def _clip_params(self, params):
        return np.clip(
            params,
            np.array([0., 1e-3, 1e-3]),
            np.array([1., 1.-1e-3, 0.20])
        )
    
    def _plot_population(self, indiv_df):
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 18
        plt.rcParams["axes.linewidth"] = 2

        fig_path = f"{self.result_path}/{self.name}/iter{self.iter:03d}/"
        for x_col in ["age", "gender"]:
            for y_sb, y_col in zip(self.param_symbol, self.targeted_params):
                if x_col == "gender":
                    fig = plt.figure(figsize=(5, 5))
                    ttest_t, ttest_p = ttest_ind(
                        indiv_df.groupby("gender").get_group("male")[y_col].to_numpy(),
                        indiv_df.groupby("gender").get_group("female")[y_col].to_numpy(),
                    )
                    gender_list = ["male", "female"]
                    palette = {"male": "#0077BB", "female": "#EE7733"}
                    sns.violinplot(
                        x=x_col,
                        y=y_col,
                        data=indiv_df.query("gender in @gender_list"),
                        order=["male", "female"],
                        palette=palette
                    )
                    plt.xlabel("Gender")
                    plt.ylabel(f"Inferred {y_sb}")
                    custom_lines = [Line2D([0], [0], color="black", lw=2)]
                    plt.legend(custom_lines, [f"$t={ttest_t:.2f}, p={ttest_p:.3f}$"], handlelength=0, handletextpad=0)
                    leg = plt.gca().get_legend()
                    leg.legendHandles[0].set_visible(False)
                else:
                    fig = plt.figure(figsize=(5, 5))
                    pearson_r, pearson_p = pearsonr(indiv_df[x_col].to_numpy(), indiv_df[y_col].to_numpy())
                    sns.regplot(
                        x=x_col,
                        y=y_col,
                        data=indiv_df,
                        scatter_kws={"s": 10, "alpha": 0.2, "color": "black"},
                        line_kws={"color": "red", "lw": 2.5}
                    )
                    plt.xlabel("Age")
                    plt.ylabel(f"Inferred {y_sb}")
                    custom_lines = [Line2D([0], [0], color="red", lw=2.5)]
                    plt.legend(custom_lines, [f"$r={pearson_r:.2f}, p={pearson_p:.3f}$"], handlelength=0.6, handletextpad=0.4)
                
                plt.grid(linestyle="--", linewidth=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(fig_path, f"user_param_distr_{x_col}_{y_col}.pdf"), dpi=300)
                plt.show()
                plt.close()
