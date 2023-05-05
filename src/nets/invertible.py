"""
Reference:
https://github.com/stefanradev93/BayesFlow
https://github.com/y0ast/Glow-PyTorch/
https://github.com/tonyduan/normalizing-flows/
"""
import numpy as np
import scipy as sp
import torch
import torch.nn as nn

from .base import NetFrame


class PermutationLayer(nn.Module):
    """
    A layer that permutes the order of input features (for Real NVP).
    """
    def __init__(self, in_sz=3):
        # Register permutation and its inverse as buffers
        super().__init__()
        self.register_buffer("permutation", torch.randperm(in_sz))
        self.register_buffer("inv_permutation", torch.argsort(self.permutation))

    def forward(self, param, inverse=False):
        """
        Apply the permutation or its inverse on the input tensor.

        param (torch.Tensor): Input tensor.
        inverse (bool, optional): If True, apply the inverse permutation, default is False.
        ---
        output (torch.Tensor): Permuted input tensor.
        """
        if inverse:
            return torch.index_select(param, -1, self.inv_permutation)
        return torch.index_select(param, -1, self.permutation)


class ActNormLayer(nn.Module):
    """
    Activation Normalization layer for Glow, performing feature-wise scaling and biasing.
    """
    def __init__(self, n_param):
        super().__init__()
        self.n_param = n_param
        self.mu = nn.Parameter(torch.zeros(n_param, dtype=torch.float))
        self.log_sigma = nn.Parameter(torch.zeros(n_param, dtype=torch.float))
        self.register_buffer("inited", torch.tensor(0, dtype=torch.bool))

    def initialize_parameters(self, input):
        """
        Initialize the parameters (mu and log_sigma) based on the input tensor.

        input (torch.Tensor): Input tensor to compute the initialization values from.
        """
        if not self.training:
            raise ValueError("ActNorm should be inited in Train mode")

        with torch.no_grad():
            # Calculate mu and log_sigma from input data
            mu = -torch.mean(input.clone(), dim=[0])
            vars = torch.mean((input.clone() + mu) ** 2, dim=[0])
            log_sigma = torch.log(1.0 / (torch.sqrt(vars) + 1e-6))

            # Update parameters
            self.mu.data.copy_(mu.data)
            self.log_sigma.data.copy_(log_sigma.data)

        self.inited = torch.tensor(1, dtype=torch.bool)

    def forward(self, param, cond=None, inverse=False):
        """
        Apply activation normalization (forward or inverse) on the input tensor.

        param (torch.Tensor): Input tensor.
        cond (torch.Tensor, optional): Conditioning tensor, currently not used.
        inverse (bool, optional): If True, apply the inverse normalization, default is False.
        ---
        outputs (tuple): A tuple containing the transformed tensor and the log determinant.
        """
        if not self.inited:
            self.initialize_parameters(param)

        if not inverse:
            z = param * torch.exp(self.log_sigma) + self.mu
            log_det = torch.sum(self.log_sigma)
            return z, log_det

        else:
            x = (param - self.mu) / torch.exp(self.log_sigma)
            log_det = -torch.sum(self.log_sigma)
            return x, log_det


class BatchNormLayer(nn.Module):
    """
    Batch normalization layer from Density estimation using Real NVP (https://arxiv.org/abs/1605.08803).
    Reference: https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
    """
    def __init__(self, num_inputs, momentum=0.1, eps=1e-5):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(num_inputs), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(num_inputs), requires_grad=True)
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.ones(num_inputs))

    def forward(self, inputs, cond=None, inverse=False, log_det_J=True):
        """
        Apply batch normalization (forward or inverse) on the input tensor.

        inputs (torch.Tensor): Input tensor.
        cond (torch.Tensor, optional): Conditioning tensor, currently not used.
        inverse (bool, optional): If True, apply the inverse normalization, default is False.
        log_det_J (bool, optional): If True, return the log determinant of the Jacobian, default is True.
        ---
        outputs (tuple): A tuple containing the transformed tensor and the log determinant.
        """
        if not inverse:
            if self.training:
                self.batch_mean = inputs.mean(dim=0)
                self.batch_var = inputs.var(dim=0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)
                self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
                self.running_var.add_(self.batch_var.data * (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            if log_det_J:
                ldj = (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)
                return y, ldj
            return y
        
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
            y = x_hat * var.sqrt() + mean
            if log_det_J:
                ldj = (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)
                return y, ldj
            return y


class Inv1x1ConvLayer(nn.Module):
    """
    1x1 Convolution for Glow
    Invertibility guaranteed by LU-decomposition
    """
    def __init__(self, n_param, LU_decomposed=True):
        super().__init__()
        self.n_param = n_param
        W = sp.linalg.qr(np.random.randn(n_param, n_param))[0]
        self.LU_decomposed = LU_decomposed

        if not self.LU_decomposed:
            self.W = nn.Parameter(torch.tensor(W, dtype=torch.float))
        else:
            P, L, U = sp.linalg.lu(W)
            self.register_buffer("P", torch.tensor(P, dtype=torch.float))
            self.L = nn.Parameter(torch.tensor(L, dtype=torch.float))
            self.S = nn.Parameter(torch.tensor(np.diag(U), dtype=torch.float))
            self.U = nn.Parameter(torch.triu(torch.tensor(U, dtype=torch.float), diagonal=1))
            self.eye = torch.eye(self.n_param)

    def forward(self, param, cond=None, inverse=False):
        """
        Apply the 1x1 convolution (forward or inverse) on the input tensor.

        param (torch.Tensor): Input tensor.
        cond (torch.Tensor, optional): Conditioning tensor, currently not used.
        inverse (bool, optional): If True, apply the inverse convolution, default is False.
        ---
        outputs (tuple): A tuple containing the transformed tensor and the log determinant.
        """
        if not inverse:
            if not self.LU_decomposed:
                z = param @ self.W
                log_det = torch.slogdet(self.W)[1]
            else:
                self.eye = self.eye.to(param.device)
                L = torch.tril(self.L, diagonal=-1) + self.eye
                U = torch.triu(self.U, diagonal=1)
                z = param @ self.P @ L @ (U + torch.diag(self.S))
                log_det = torch.sum(torch.log(torch.abs(self.S)))
            return z, log_det
        else:
            if not self.LU_decomposed:
                x = param @ torch.inverse(self.W)
                log_det = -torch.slogdet(self.W)[1]
            else:
                self.eye = self.eye.to(param.device)
                L = torch.tril(self.L, diagonal=-1) + self.eye
                U = torch.triu(self.U, diagonal=1)
                W = self.P @ L @ (U + torch.diag(self.S))
                x = param @ torch.inverse(W)
                log_det = -torch.sum(torch.log(torch.abs(self.S)))
            return x, log_det


class ConditionalLayer(nn.Module):
    """
    A conditional layer that combines input tensor and conditioning tensor.
    """
    def __init__(self, in_sz, out_sz, config):
        super().__init__()
        head_seq = []
        for i in range(config["head_depth"]):
            layer_in = config["cond_sz"]
            layer_out = config["head_sz"] if i == config["head_depth"] - 1 else config["cond_sz"]
            head_seq.append(nn.Linear(layer_in, layer_out))
            if i < config["depth"] - 1:
                head_seq.append(nn.Tanh())
            self.head_layers = nn.Sequential(*head_seq)

        seq = []
        first_in_sz = in_sz
        first_in_sz += config["head_sz"] if config["head_depth"] > 0 else config["cond_sz"]
        for i in range(config["depth"]):
            layer_in = first_in_sz if i == 0 else config["feat_sz"]
            layer_out = out_sz if i == config["depth"] - 1 else config["feat_sz"]
            seq.append(nn.Linear(layer_in, layer_out))
            if i < config["depth"] - 1:
                seq.append(nn.Tanh())

        self.layers = nn.Sequential(*seq)

    def forward(self, param, cond):
        """
        Apply the conditional layer on the input tensor.

        param (torch.Tensor): Input tensor.
        cond (torch.Tensor): Conditioning tensor.
        ---
        outputs (torch.Tensor): Transformed tensor.
        """
        cond = self.head_layers(cond)
        x = torch.cat([param, cond], dim=-1)
        return self.layers(x)


class ConditionalAffineBlock(nn.Module):
    """
    A conditional affine block for normalizing flows.
    """
    def __init__(self, n_param, config):
        super().__init__()
        self.sz1 = n_param // 2
        self.sz2 = n_param - self.sz1

        if config["permutation"]:
            self.permutation = PermutationLayer(n_param)
        else:
            self.permutation = None
        
        self.s1 = ConditionalLayer(self.sz2, self.sz1, config)
        self.t1 = ConditionalLayer(self.sz2, self.sz1, config)
        self.s2 = ConditionalLayer(self.sz1, self.sz2, config)
        self.t2 = ConditionalLayer(self.sz1, self.sz2, config)

    def forward(self, param, cond, inverse=False, log_det_J=True):
        """
        Apply the conditional affine block (forward or inverse) on the input tensor.

        param (torch.Tensor): Input tensor.
        cond (torch.Tensor): Conditioning tensor.
        inverse (bool, optional): If True, apply the inverse transformation, default is False.
        log_det_J (bool, optional): If True, return the log determinant of the Jacobian, default is True.
        ---
        outputs (tuple): A tuple containing the transformed tensor and the log determinant.
        """
        if not inverse:
            if self.permutation is not None:
                param = self.permutation(param)

            u1, u2 = param.split([self.sz1, self.sz2], dim=-1)
            s1 = self.s1(u2, cond)
            v1 = u1 * torch.exp(s1) + self.t1(u2, cond)
            s2 = self.s2(v1, cond)
            v2 = u2 * torch.exp(s2) + self.t2(v1, cond)
            v = torch.cat([v1, v2], dim=-1)

            if log_det_J:
                return v, s1.sum(dim=-1) + s2.sum(dim=-1)
            return v

        else:
            v1, v2 = param.split([self.sz1, self.sz2], dim=-1)
            s2 = self.s2(v1, cond)
            u2 = (v2 - self.t2(v1, cond)) * torch.exp(-s2)
            s1 = self.s1(u2, cond)
            u1 = (v1 - self.t1(u2, cond)) * torch.exp(-s1)
            u = torch.cat([u1, u2], dim=-1)
            
            if self.permutation is not None:
                u = self.permutation(u, inverse=True)
                
            if log_det_J:
                return u, -s1.sum(dim=-1) - s2.sum(dim=-1)
            return u


class InvertibleNet(NetFrame):
    """
    An invertible network for normalizing flows.
    This class supports following different types of flows:

    i) Glow: a block includes (act_norm, inv_1x1_conv, affine_block)
    ii) Real NVP: a block includes (affine_block_with_permutation, batch_norm)
    """
    def __init__(self, config):
        super().__init__(config)
        n_block = config["n_block"]
        self.n_param = config["param_sz"]
        modules = list()
        for i in range(n_block):
            if config["act_norm"]:
                modules += [ActNormLayer(self.n_param)]
            if config["invert_conv"]:
                modules += [Inv1x1ConvLayer(self.n_param, LU_decomposed=True)]
            modules += [ConditionalAffineBlock(self.n_param, config["block"])]
            if config["batch_norm"]:
                modules += [BatchNormLayer(self.n_param)]
        self.cINNs = nn.ModuleList(modules)
        self.compute_total_params("invertible")

    def forward(self, param, cond, inverse=False):
        """
        Apply the invertible network (forward or inverse) on the input tensor.

        param (torch.Tensor): Input tensor.
        cond (torch.Tensor): Conditioning tensor.
        inverse (bool, optional): If True, apply the inverse transformation, default is False.
        ---
        outputs (tuple): A tuple containing the transformed tensor and the log determinant.
        """
        log_det_Js = []
        if not inverse:
            z = param
            for cINN in self.cINNs:
                z, log_det_J = cINN(z, cond)
                log_det_Js.append(log_det_J)
            log_det_J = sum(log_det_Js)
            return z, log_det_J
        else:
            theta = param
            for cINN in reversed(self.cINNs):
                theta, log_det_J = cINN(theta, cond, inverse=True)
                log_det_Js.append(log_det_J)
            log_det_J = sum(log_det_Js)
            return theta, log_det_J

    def sample(self, cond, n_sample):
        """
        Generate samples from the invertible network given the conditioning tensor.

        cond (torch.Tensor): Conditioning tensor.
        n_sample (int): Number of samples to generate.
        ---
        outputs (tuple): A tuple containing the sampled tensor and the log determinant.
        """
        # In case observed data is a single trial
        if int(cond.shape[0]) == 1:
            z_sampled = torch.randn((n_sample, self.n_param)).to(self.device)
            param_sampled, log_det_J = self.forward(
                z_sampled,
                cond.repeat((n_sample, 1)),
                inverse=True
            )
        # In case observed data is a batch of trials
        else:
            z_sampled = torch.randn((n_sample, int(cond.shape[0]), self.n_param)).to(self.device)
            param_sampled, log_det_J = self.forward(
                z_sampled,
                torch.stack([cond] * n_sample),
                inverse=True
            )
        return param_sampled, log_det_J