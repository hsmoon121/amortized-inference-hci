import numpy as np
import torch
from einops import rearrange
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar

from .base import NetFrame
from .encoder import EncoderNet, TrialAttentionNet, TrialInvariantNet
from .invertible import InvertibleNet
from .utils import mask_and_pad_traj_data, sort_and_pad_traj_data, get_auto_device


class AmortizerFrame(NetFrame):
    def __init__(self, config):
        super().__init__(config)

    def _set_device(self, config):
        """
        Set device for the model. 
        If device is not specified in the config, it will be automatically set.

        config (dict): Configuration dictionary for the amortizer model.
        """
        if "device" not in config:
            self.given_device = get_auto_device()
        elif config["device"] is None:
            self.given_device = get_auto_device()
        else:
            self.given_device = torch.device(config["device"])
        self.to(self.given_device)

    def infer(self, stat_data, traj_data=None, n_sample=100, type="mode", return_samples=False):
        """
        Infer the posterior distribution of the parameters given the static and trajectory data.

        stat_data (ndarray): array of static data with a shape (n_batch, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (traj_length, traj_sz).
        n_sample (int, optional): Number of posterior samples, default is 100.
        type (str, optional): Type of inference from distribution ("mode", "mean", "median"), default is "mode".
        return_samples (bool, optional): Whether to return posterior samples, default is False.
        ---
        outputs (tuple): Tuple containing inferred parameters (torch.Tensor) and posterior samples (torch.Tensor).
        """
        n_param = self.invertible_net.n_param
        post_sampled, _ = self._sample(stat_data, traj_data, n_sample)

        if type == "mode": # Maximum a Posteriori (MAP) estimation
            results = []
            if len(post_sampled.shape) > 2:
                # If the posterior is multi-modal (>2), use the maximum of the product of the marginal densities
                x = np.linspace(-3., 3., 300)
                for i in range(n_param):
                    kdes = [gaussian_kde(post_sampled[:, tr, i]) for tr in range(post_sampled.shape[1])]
                    y = np.array([np.prod([kde(xi) for kde in kdes]) for xi in x])
                    results.append(x[np.argmax(y)])
            else:
                # Use scipy.optimize.minimize_scalar to find the mode
                for i in range(n_param):
                    kde = gaussian_kde(post_sampled[:, i])
                    opt_x = minimize_scalar(lambda x: -kde(x), method="golden").x
                    if hasattr(opt_x, "__len__"):
                        results.append(opt_x[0])
                    else:
                        results.append(opt_x)
            results = np.array(results)

        elif type == "mean":
            post_sampled = post_sampled.reshape((-1, n_param))
            results = np.mean(post_sampled, axis=0)

        elif type == "median":
            post_sampled = post_sampled.reshape((-1, n_param))
            results = np.median(post_sampled, axis=0)
        else:
            raise Exception(f"inappropriate type: {type}")
        
        if return_samples:
            return results, post_sampled
        else:
            return results


class AmortizerForSummaryData(AmortizerFrame):
    def __init__(self, config=dict()):
        """
        Amortizer model for summary data (i.e., fixed-size data per inference)

        config (dict): Configuration dictionary for the amortizer model.
        """
        super().__init__(config)
        self.encoder_net = EncoderNet(config["encoder"])
        self.invertible_net = InvertibleNet(config["invertible"])
        self._set_device(config)

    def forward(self, params, stat_data, traj_data=None):
        """
        params (ndarray): array of parameters with a shape (n_batch, n_param).
        stat_data (ndarray): array of static data with a shape (n_batch, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (traj_length, traj_sz).
        ---
        outputs (list): List of outputs from the invertible network.
        """
        if not self.encoder_net.series_data:
            params = torch.FloatTensor(params).to(self.device)
            cond = self.encoder_net(stat_data)
            out = self.invertible_net(params, cond)

        elif self.encoder_net.traj_encoder_type == "transformer":
            padded_traj, mask = mask_and_pad_traj_data(traj_data)
            params = torch.FloatTensor(params).to(self.device)
            cond = self.encoder_net(stat_data, padded_traj, mask=mask)
            out = self.invertible_net(params, cond)
            
        else:
            sorted_args = sort_and_pad_traj_data(stat_data, traj_data)
            params = torch.FloatTensor(params)[sorted_args[-1]].to(self.device)
            cond = self.encoder_net(*sorted_args[0:-1])
            _, inv_index = sorted_args[-1].sort()
            out = self.invertible_net(params, cond)
            out = [o[inv_index] for o in out]
        return out

    def _sample(self, stat_data, traj_data=None, n_sample=100):
        """
        Sample from the posterior distribution of the parameters given the static and trajectory data.

        stat_data (ndarray): array of static data with a shape (n_batch, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (traj_length, traj_sz).
        n_sample (int, optional): Number of posterior samples, default is 100.
        ---
        outputs (tuple): Tuple containing posterior samples (ndarray) and log determinant of Jacobian (ndarray).
        """
        if not self.encoder_net.series_data:
            cond = self.encoder_net(stat_data)
        elif self.encoder_net.traj_encoder_type == "transformer":
            padded_traj, mask = mask_and_pad_traj_data(traj_data)
            cond = self.encoder_net(stat_data, padded_traj, mask=mask)
        else:
            sorted_args = sort_and_pad_traj_data(stat_data, traj_data)
            cond = self.encoder_net(*sorted_args[0:-1])
        args = self.invertible_net.sample(cond, n_sample)
        post_sampled, log_det_J = [arg.cpu().detach().numpy() for arg in args]
        return post_sampled, log_det_J


class AmortizerForTrialData(AmortizerFrame):
    def __init__(self, config=dict()):
        """
        Amortizer model for full data from multiple i.i.d. trials (i.e., variable-size data per inference)

        config (dict): Configuration dictionary for the amortizer model.
        """
        super().__init__(config)

        self.encoder_net = EncoderNet(config["encoder"])

        if config["trial_encoder_type"] == "attention":
            self.trial_encoder_net = TrialAttentionNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["attention"],
            )
        elif config["trial_encoder_type"] == "invariant":
            self.trial_encoder_net = TrialInvariantNet(
                self.encoder_net.compute_output_sz(),
                config["trial_encoder"]["invariant"],
            )
        else:
            raise RuntimeError("trial_encoder_type should be among 'attention' and 'invariant'.")

        self.invertible_net = InvertibleNet(config["invertible"])
        self._set_device(config)

    def forward(self, batch_param, batch_stat, batch_traj=None):
        """
        batch_param (ndarray): array of parameters with a shape (n_batch, n_param).
        batch_stat (ndarray): array of static data with a shape (n_batch, n_trial, stat_sz).
        batch_traj (list, optional): List of trajectory data, each item should have a shape (n_trial, traj_length, traj_sz).
        ---
        outputs (list): List of outputs from the invertible network.
        """
        full_stat_data = rearrange(batch_stat, "b e s -> (b e) s")

        if not self.encoder_net.series_data:
            contexts = self.encoder_net(full_stat_data)
            contexts = rearrange(contexts, "(b e) c -> b e c", b=batch_param.shape[0])
            
        else:
            full_stat_data = rearrange(batch_stat, "b e s -> (b e) s")
            full_traj_data = list()
            for data in batch_traj:
                full_traj_data += list(data)

            if self.encoder_net.traj_encoder_type == "transformer":
                padded_traj, mask = mask_and_pad_traj_data(full_traj_data)
                contexts = self.encoder_net(full_stat_data, padded_traj, mask=mask)
                contexts = rearrange(contexts, "(b e) c -> b e c", b=batch_param.shape[0])
            else:
                sorted_args = sort_and_pad_traj_data(full_stat_data, full_traj_data)
                sorted_contexts = self.encoder_net(*sorted_args[0:-1])
                _, inv_index = sorted_args[-1].sort()
                contexts = sorted_contexts[inv_index]
                contexts = rearrange(contexts, "(b e) c -> b e c", b=batch_param.shape[0])

        batch_param = torch.FloatTensor(batch_param).to(self.device)
        cond = self.trial_encoder_net(contexts)
        out = self.invertible_net(batch_param, cond)
        return out

    def _sample(self, stat_data, traj_data=None, n_sample=100):
        """
        Sample from the posterior distribution of the parameters given the static and trajectory data.

        stat_data (ndarray): array of static data with a shape (n_batch, n_trial, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (n_trial, traj_length, traj_sz).
        n_sample (int, optional): Number of posterior samples, default is 100.
        ---
        outputs (tuple): Tuple containing posterior samples (ndarray) and log determinant of Jacobian (ndarray).
        """
        if not self.encoder_net.series_data:
            contexts = self.encoder_net(stat_data)
        elif self.encoder_net.traj_encoder_type == "transformer":
            padded_traj, mask = mask_and_pad_traj_data(traj_data)
            contexts = self.encoder_net(stat_data, padded_traj, mask=mask)
        else:
            sorted_args = sort_and_pad_traj_data(stat_data, traj_data)
            contexts = self.encoder_net(*sorted_args[0:-1])

        cond = self.trial_encoder_net(contexts)
        args = self.invertible_net.sample(cond, n_sample)
        post_sampled, log_det_J = [arg.cpu().detach().numpy() for arg in args]
        return post_sampled, log_det_J