import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def MaskedGlobalMaxPool1D(x, mask):
    """
    x: (batch_size, channels, length) Tensor
    mask: (batch_size, channels, length) Tensor that consists of 1 or 0
    ---
    output: (batch_size, channels) Tensor
    """
    x[mask == 0] = -float("Inf")
    return x.max(-1)[0]


def MaskedGlobalAvgPool1D(x, mask):
    """
    x: (batch_size, channels, length) Tensor
    mask: (batch_size, channels, length) Tensor that consists of 1 or 0
    ---
    output: (batch_size, channels) Tensor
    """
    return (x * mask).sum(-1) / mask.sum(-1)


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_sz,
        out_sz,
        hidden_sz=512,
        hidden_depth=1,
        batch_norm=False,
        activation="relu"
    ):
        """
        in_sz: int, input size
        out_sz: int, output size
        hidden_sz: int, hidden size (default: 512)
        hidden_depth: int, number of hidden layers (default: 1)
        batch_norm: bool, whether to use batch normalization (default: False)
        activation: str, activation function (default: "relu")
        """
        super().__init__()
        layers = []
        for i in range(hidden_depth + 1):
            layer_in = in_sz if i == 0 else hidden_sz
            layer_out = out_sz if i == hidden_depth else hidden_sz
            layers.append(nn.Linear(layer_in, layer_out))

            if batch_norm and i != hidden_depth:
                layers.append(nn.BatchNorm1d(layer_out))
            if i != hidden_depth:
                if activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                else:
                    pass
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch_size, in_sz) Tensor
        ---
        output: (batch_size, out_sz) Tensor
        """
        return self.layers(x)

            
class Conv1dBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        depth,
        kernel_sz,
        global_pool=None,
        batch_norm=True,
        bn_first=True
    ):
        """
        ch_in: int, number of input channels
        ch_out: int, number of output channels
        depth: int, depth of convolutional layers
        kernel_sz: int, kernel size
        global_pool: str, pooling type ("max" or "avg") if global pooling is needed (default: None)
        batch_norm: bool, whether to use batch normalization (default: True)
        bn_first: bool, whether to apply batch normalization before activation (default: True)
        """
        super().__init__()
        self.depth = depth
        self.global_pool = global_pool
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        conv_ls, bn_ls = list(), list()
        for i in range(depth):
            if i == 0:
                conv_ls.append(nn.Conv1d(ch_in, ch_out, kernel_sz, padding="same", dilation=1))
            else:
                conv_ls.append(nn.Conv1d(ch_out, ch_out, kernel_sz, padding="same", dilation=2))
            if self.batch_norm:
                bn_ls.append(nn.BatchNorm1d(ch_out))

        self.conv_layers = nn.ModuleList(conv_ls)
        if self.batch_norm:
            self.bn_layers = nn.ModuleList(bn_ls)

    def forward(self, x, padded=None):
        """
        x: (batch_size, length, ch_in) Tensor
        padded: list of sizes of zero-paddings (optional)
        ---
        output: (batch_size, ch_out) Tensor if pooling is used, otherwise (batch_size, length, ch_out) Tensor
        """
        x = x.transpose(-1, -2)

        for i in range(self.depth):
            x = self.conv_layers[i](x)
            if self.batch_norm and self.bn_first:
                x = self.bn_layers[i](x)
            x = F.gelu(x)
            if self.batch_norm and not self.bn_first:
                x = self.bn_layers[i](x)

        mask = torch.ones(x.shape).to(x.device)
        if padded is not None:
            for i in range(len(padded)):
                mask[i, :, mask.size(1) - padded[i]:] = 0.

        if self.global_pool == "max":
            x = MaskedGlobalMaxPool1D(x, mask)
        elif self.global_pool == "avg":
            x = MaskedGlobalAvgPool1D(x, mask)
        else:
            x = x * mask
            x = x.transpose(-1, -2)
        return x

        
class NetFrame(nn.Module):
    def __init__(self, config=None):
        super().__init__()

    def compute_total_params(self, str="amortizer"):
        params = 0
        for p in list(self.parameters()):
            params += np.prod(list(p.size()))
        self.total_params = params
        print(f"[ {str} ] total trainable parameters : {self.total_params}")

    @property
    def device(self):
        if len(list(self.parameters())) > 0:
            return next(self.parameters()).device
        else:
            return None