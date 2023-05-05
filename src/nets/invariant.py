"""
Reference:
https://github.com/stefanradev93/BayesFlow/blob/master/bayesflow/networks.py
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat

from .base import LinearBlock

class InvariantModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_sz = config["in_sz"]
        feat_sz = config["feat_sz"]
        depth = config["depth"]
        self.s1 = LinearBlock(in_sz, feat_sz, feat_sz, hidden_depth=depth)
        self.s2 = LinearBlock(feat_sz, in_sz, feat_sz, hidden_depth=depth)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, n_obs, in_dim)
        ---
        outputs: Tensor of shape (batch_size, in_dim)
        """
        x_reduced = torch.mean(self.s1(x), dim=1)
        out = self.s2(x_reduced)
        return out
    
    
class EquivariantModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_sz = config["in_sz"]
        feat_sz = config["feat_sz"]
        depth = config["depth"]
        self.inv_module = InvariantModule(config)
        self.s3 = LinearBlock(in_sz * 2, in_sz, feat_sz, hidden_depth=depth)
                    
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, n_obs, in_dim)
        ---
        outputs: Tensor of shape (batch_size, n_obs, in_dim)
        """
        # (batch_size, N, in_dim) -> (batch_size, in_dim) -> (batch_size, N, in_dim)
        N = int(x.shape[1])
        out_inv = self.inv_module(x)
        out_inv_rep = repeat(out_inv, "b f -> b n f", n=N)
        
        # (batch_size, N, in_dim * 2) -> (batch_size, N, in_dim)
        out_cat = torch.cat((x, out_inv_rep), dim=-1)
        out = self.s3(out_cat)
        return out