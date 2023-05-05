import os
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
import torch

from ..utils import Logger
from ..utils.plot import plot_parameter_recovery, pair_plot
from ..utils.distrib_distance import hist_kld, mmd_kernel


class BaseTrainer(ABC):
    def __init__(self, name="sample", task_name="sample"):
        self.iter = 0
        self.name = name
        self.task_name = task_name
        self.amortizer = None
        self.simulator = None
        self.optimizer = None
        self.scheduler = None
        self.targeted_params = None
        self.param_symbol = None
        self.base_params = None
        self.obs_label = None
        self.obs_description = None

        self.model_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "data",
            f"{self.task_name}",
            "amortizer_models"
        )
        self.board_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "data",
            f"{self.task_name}",
            "board"
        )
        self.result_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "results",
            f"{self.task_name}"
        )
        self.clipping = float("Inf")

    def train(
        self,
        n_iter,
        step_per_iter,
        batch_sz,
        n_trial=100,
        train_mode="replay",
        capacity=10000,
        board=True
    ):
        """
        Training loop

        n_iter (int): Number of training iterations
        step_per_iter (int): Number of training steps per iteration
        batch_sz (int): Batch size
        n_trial (int): Number of trials for each user data (default: 1)
        train_mode (str): Training mode among "online", "offline", "replay" (default: "replay")
        capacity (int): Replay buffer capacity (default: 10000)
        board (bool): Whether to use tensorboard (default: True)
        """
        iter = self.iter
        last_step = self.iter * step_per_iter

        self._set_train_mode(train_mode, capacity)
        self.logger = Logger(self.name, last_step, board=board, board_path=self.board_path)

        # Training iterations
        losses = dict()
        print(f"\n[ Training - {self.name}]")
        for iter in range(self.iter + 1, n_iter + 1):
            losses[iter] = []

            # Training loop
            with tqdm(total=step_per_iter, desc=f" Iter {iter}") as progress:
                for step in range(step_per_iter):
                    if train_mode == "online":
                        batch_args = self._get_online_batch(batch_sz, n_trial)
                    elif train_mode == "offline":
                        batch_args = self._get_offline_batch(batch_sz, n_trial)
                    else:
                        batch_args = self._get_replay_batch(batch_sz, n_trial)

                    # Training step
                    loss = self._train_step(*batch_args)
                    losses[iter].append(loss)

                    # Logging
                    if step % 10 == 0:
                        self.logger.write_scalar(train_loss=loss, lr=self.scheduler.get_last_lr()[0])
                    progress.set_postfix_str(f"Avg.Loss: {np.mean(losses[iter]):.3f}")
                    progress.update(1)
                    self.logger.step()
                    self.scheduler.step((iter-1) + step/step_per_iter)
                    
                    if np.isnan(loss):
                        raise RuntimeError("Nan loss computed.")

            # Save model
            if iter % 10 == 0:
                self.save(iter)
            self.iter = iter

        print("\n[ Training Done ]")
        if iter in losses:
            print(f"  Training Loss: {np.mean(losses[iter])}\n")

    def _set_train_mode(self, train_mode, capacity):
        pass

    def _get_online_batch(self, batch_sz, n_trial):
        pass

    def _get_offline_batch(self, batch_sz, n_trial):
        pass

    def _get_replay_batch(self, batch_sz, n_trial):
        pass

    def _clip_params(self, params):
        pass

    def _behavior_bin_info(self, label):
        pass

    def _train_step(self, params, stat_data, traj_data=None):
        """
        Training step

        params (ndarray): [n_sim, n_param] array of parameters
        stat_data (list): [n_sim] list of static data
        traj_data (list): [n_sim] list of trajectories (default: None)
        """
        self.amortizer.train()
        z, log_det_J = self.amortizer(params, stat_data, traj_data)
        loss = torch.mean(0.5 * torch.square(torch.norm(z, dim=-1)) - log_det_J)
        return self._optim_step(loss)
    
    def _optim_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.amortizer.parameters():
            param.grad.data.clamp_(-self.clipping, self.clipping)
        self.optimizer.step()
        return loss.item()

    @abstractmethod
    def valid(self, *args, **kwargs):
        pass

    def save(self, iter, path=None):
        """
        Save model, optimizer, and scheduler with iteration number
        """
        if path is None:
            os.makedirs(f"{self.model_path}/{self.name}", exist_ok=True)
            ckpt_path = f"{self.model_path}/{self.name}/iter{iter:03d}.pt"
        else:
            os.makedirs(path, exist_ok=True)
            ckpt_path = path + f"iter{iter:03d}.pt"
        torch.save({
            "iteration": iter,
            "model_state_dict": self.amortizer.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, ckpt_path)
        
    def load(self):
        """
        Load model, optimizer, and scheduler from the latest checkpoint
        """
        ckpt_path = self._find_last_ckpt(f"{self.model_path}/{self.name}")
        if ckpt_path is None:
            print(f"[ amortizer - no checkpoint ] start training from scratch.")
            iter = 0
        else:
            ckpt = torch.load(ckpt_path, map_location=self.amortizer.device.type)
            self.amortizer.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optim_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"[ amortizer - loaded checkpoint ]\n\t{ckpt_path}")
            iter = ckpt["iteration"]
            self.scheduler.step(iter)
        self.iter = iter

    def _find_last_ckpt(self, save_path):
        """
        Find the latest checkpoint
        """
        os.makedirs(save_path, exist_ok=True)
        ckpts = os.listdir(save_path)
        ckpts_with_iter = [f for f in ckpts if f.startswith("iter")]
        if ckpts_with_iter:
            return os.path.join(save_path, max(ckpts_with_iter))
        elif ckpts:
            return os.path.join(save_path, max(ckpts))
        else:
            return None

    def _plot_parameter_recovery(
        self,
        y_true,
        y_pred,
        y_fit,
        r_squared,
        fname,
        param_label=None,
        fpath=None
    ):
        if fpath is not None:
            fig_path = fpath
        else:
            fig_path = f"{self.result_path}/{self.name}/iter{self.iter:03d}/"

        plot_parameter_recovery(
            y_true,
            y_pred,
            y_fit,
            r_squared,
            fname,
            param_label=param_label,
            fpath=fig_path
        )

    def _pair_plot(
        self,
        sampled_params,
        param_labels,
        limits=None,
        gt_params=None,
        fname="sample",
        fpath=None,
    ):
        if fpath is not None:
            fig_path = fpath
        else:
            fig_path = f"{self.result_path}/{self.name}/iter{self.iter:03d}/"
        
        pair_plot(
            sampled_params,
            param_labels,
            limits,
            gt_params,
            fname,
            fpath=fig_path
        )

    def parameter_recovery(
        self,
        res,
        gt_params,
        valid_data,
        n_sample,
        infer_type,
        plot=False,
        surfix="",
    ):
        n_param = gt_params.shape[0]
        inferred_params = list()

        for param_i in range(n_param):
            if self.task_name == "pnc":
                # Special case of point-and-click: Using trajectory data & log-normal values of params
                stat_i, traj_i = valid_data[param_i]
                lognorm_param = self.amortizer.infer(stat_i, traj_i, n_sample=n_sample, type=infer_type)
                lognorm_param = self._clip_params(lognorm_param)
                inferred_params.append(self.simulator.convert_from_output(lognorm_param)[0])
            else:
                if len(valid_data[param_i].shape) == 1: # Summary data case
                    stat_i = np.expand_dims(valid_data[param_i], axis=0)
                else: # Trial data case
                    stat_i = valid_data[param_i]
                inferred_param = self.amortizer.infer(stat_i, n_sample=n_sample, type=infer_type)
                inferred_params.append(self._clip_params(inferred_param))
        inferred_params = np.array(inferred_params)
        
        for i, l in enumerate(self.targeted_params):
            y_true = gt_params[:, i]
            y_pred = inferred_params[:, i]

            if np.std(y_true) == 0:
                # In case there are no ground-truth values (i.e., all values are 0)
                pass
            else:
                y_fit = np.polyfit(y_true, y_pred, 1)
                y_func = np.poly1d(y_fit)
                r_squared = r2_score(y_pred, y_func(y_true))

                res["Parameter_Recovery/r2_" + l + surfix] = r_squared
                if plot:
                    self._plot_parameter_recovery(
                        y_true,
                        y_pred,
                        y_fit,
                        r_squared,
                        fname=f"r2_" + l + surfix,
                        param_label=self.param_symbol[i],
                    )

    def _plot_histogram(self, behavior, fname, color="g", maxyticks=5):
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 16
        plt.rcParams["axes.linewidth"] = 2

        n_column = len(behavior)
        fig = plt.figure(figsize=(5, 1.8 * n_column))

        for idx, column in enumerate(behavior):
            minbin, maxbin, nbins = self._behavior_bin_info(column)
            bins, bin_edges = np.histogram(
                behavior[column],
                bins=nbins,
                range=(minbin, maxbin),
            )
            bins = bins / sum(bins)
            ax = plt.subplot(n_column, 1, idx + 1)
            width = bin_edges[1] - bin_edges[0]
            label = "m={:.2f} std={:.2f}".format(np.mean(behavior[column]), np.std(behavior[column]))
            plt.bar(bin_edges[:-1], bins, width=width, color=color, edgecolor="black", linewidth=1.0)
            plt.xlim(min(bin_edges) - width * 0.5, max(bin_edges) + width * 0.5)

            deltaticks = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
            yticks = None
            for dt in deltaticks:
                if not max(bins) > 0.0:
                    break
                yticks = np.arange(0, (int(max(bins) / dt) + 2) * dt, dt)
                if len(yticks) <= maxyticks:
                    if ax is not None: ax.set_yticks(yticks)
                    else: plt.yticks(yticks)
                    break

            plt.title(self.obs_description[idx])
            custom_lines = [Line2D([0], [0], color="black", lw=2)]
            plt.legend(custom_lines, [label,], handlelength=0, handletextpad=0)
            leg = plt.gca().get_legend()
            leg.legendHandles[0].set_visible(False)

        fig.suptitle(fname)
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        fig.subplots_adjust(top=0.88)
        fig_path = f"{self.result_path}/{self.name}/iter{self.iter:03d}/"
        plt.savefig(os.path.join(fig_path, fname + ".pdf"), dpi=300)
        plt.show()
        plt.close(fig)

    def behavior_distance(self, obs, sim, metrics, label="base"):
        keys, values = list(), list()

        # Calculate three distances for each behavior metric
        for metric in metrics:
            # Mean distance
            keys.append(f"Behavior_Distance/abs_{metric}_{label}")
            values.append(abs(np.mean(obs[metric]) - np.mean(sim[metric])))

            # KL divergence
            keys.append(f"Behavior_Distance/kl_{metric}_{label}")
            values.append(hist_kld(np.array(obs[metric]), np.array(sim[metric])))
            
            # MMD distance
            keys.append(f"Behavior_Distance/mmd_{metric}_{label}")
            values.append(mmd_kernel(
                np.array(obs[metric]).reshape((-1, 1)),
                np.array(sim[metric]).reshape((-1, 1))
            ).item())
        return keys, values