import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Reduce

from .attention import AttentionBlock
from .invariant import InvariantModule, EquivariantModule
from .base import NetFrame, Conv1dBlock
from .utils import fourier_encode


class TransformerModule(NetFrame):
    def __init__(self, in_sz, trans_config):
        """
        Transformer module for processing sequential data (e.g., trajectory).

        in_sz (int): Input size.
        trans_config (dict): Configuration dictionary for the transformer module.
        """
        super().__init__()
        num_latents = trans_config["num_latents"]
        n_block = trans_config["n_block"]
        latent_sz = trans_config["query_sz"]

        self.out_sz = trans_config["out_sz"]
        self.max_freq = trans_config["max_freq"]
        self.n_freq_bands = trans_config["n_freq_bands"]
        self.max_step = trans_config["max_step"]

        trans_config["context_sz"] = in_sz + self.n_freq_bands*2 + 1

        self.trans_latents = nn.Parameter(torch.randn(num_latents, latent_sz))
        self.trans_blocks = nn.Sequential(*[AttentionBlock(trans_config) for _ in range(n_block)])
        self.trans_head = nn.Sequential(
            Reduce("b n d -> b d", "mean"),
            nn.LayerNorm(latent_sz),
            nn.Linear(latent_sz, self.out_sz)
        )

    def forward(self, x_seq, padded=None, mask=None):
        """
        x_seq (torch.Tensor): Input sequence tensor with shape (n_batch, data_length, feat_sz).
        padded (optional): List of sizes of zero-paddings (not used in this implementation).
        mask (optional): Mask tensor for the transformer module.
        ---
        output (torch.Tensor): Output tensor with shape (n_batch, out_sz).
        """
        n, l = x_seq.shape[0], x_seq.shape[1]
        full_pos = torch.linspace(-1., 1., steps=self.max_step, device=self.device, dtype=x_seq.dtype)
        enc_pos = fourier_encode(full_pos[:l], self.max_freq, self.n_freq_bands)
        enc_pos = repeat(enc_pos, "... -> n ...", n=n)
        x_seq = torch.cat((x_seq, enc_pos), dim=-1)

        latent = repeat(self.trans_latents, "l c -> n l c", n=n)
        mask = mask.to(self.device)
        for block in self.trans_blocks:
            latent = block(latent, x_seq, mask=mask)
        x_seq = self.trans_head(latent)
        return x_seq


class ConvPoolingModule(NetFrame):
    def __init__(self, in_sz, conv_config, batch_norm=True):
        """
        Convolutional NN with global pooling module for processing sequential data.

        in_sz (int): Input size.
        conv_config (dict): Configuration dictionary for the convolutional module.
        batch_norm (bool, optional): Whether to use batch normalization, default is True.
        """
        super().__init__()
        self.batch_norm = batch_norm
        self.out_sz = 0

        n_conv = len(conv_config)
        conv_ls = list()
        for i in range(n_conv):
            conv_ls.append(Conv1dBlock(
                in_sz, 
                conv_config[i]["ch_out"],
                conv_config[i]["depth"],
                conv_config[i]["kernel_sz"],
                global_pool=conv_config[i]["global_pool"],
                batch_norm=self.batch_norm,
            ))
            self.out_sz += conv_config[i]["ch_out"]
        self.conv_layers = nn.ModuleList(conv_ls)

    def forward(self, x_seq, padded=None, mask=None):
        """
        x_seq (torch.Tensor): Input sequence tensor with shape (n_batch, data_length, feat_sz).
        padded (optional): List of sizes of zero-paddings.
        mask (optional): Mask tensor (not used in this implementation).
        ---
        output (torch.Tensor): Output tensor with shape (n_batch, out_sz).
        """
        conv_outputs = list()
        for conv in self.conv_layers:
            conv_outputs.append(conv(x_seq, padded))
        x_seq = torch.cat(conv_outputs, dim=-1)
        return x_seq


class ConvRnnModule(NetFrame):
    def __init__(self, in_sz, conv_config, rnn_config, batch_norm=True):
        """
        1-D Convolution + RNN module for processing sequential data.

        in_sz (int): Input size.
        conv_config (dict): Configuration dictionary for the convolutional module.
        rnn_config (dict): Configuration dictionary for the RNN module.
        batch_norm (bool, optional): Whether to use batch normalization, default is True.
        """
        super().__init__()
        self.batch_norm = batch_norm
        self.out_sz = rnn_config["feat_sz"]
        if self.rnn_bidirectional:
            self.out_sz *= 2

        # Conv1d part
        n_conv = len(conv_config)
        conv_feat_sz = 0
        conv_ls = list()

        # Multiple CNN kernels can be applied based on given list.
        for i in range(n_conv):
            conv_ls.append(Conv1dBlock(
                in_sz, 
                conv_config[i]["ch_out"],
                conv_config[i]["depth"],
                conv_config[i]["kernel_sz"],
                global_pool=None,
                batch_norm=self.batch_norm,
            ))
            conv_feat_sz += conv_config[i]["ch_out"]
        self.conv_layers = nn.ModuleList(conv_ls)

        # Rnn part
        self.rnn_type = rnn_config["type"]
        self.rnn_bidirectional = rnn_config["bidirectional"]
        rnn_func = nn.LSTM if rnn_config["type"] == "LSTM" else nn.GRU
        self.rnn_layers = rnn_func(
            conv_feat_sz,
            rnn_config["feat_sz"],
            rnn_config["depth"],
            batch_first=True,
            dropout=rnn_config["dropout"],
            bidirectional=self.rnn_bidirectional,
        )
        if self.batch_norm:
            self.rnn_bn = nn.BatchNorm1d(self.out_sz)

    def forward(self, x_seq, padded=None, mask=None):
        """
        x_seq (torch.Tensor): Input sequence tensor with shape (n_batch, data_length, feat_sz).
        padded (optional): List of sizes of zero-paddings.
        mask (optional): Mask tensor (not used in this implementation).
        ---
        output (torch.Tensor): Output tensor with shape (n_batch, out_sz).
        """
        conv_outputs = list()
        for conv in self.conv_layers:
            conv_outputs.append(conv(x_seq, padded))
        x_seq = torch.cat(conv_outputs, dim=-1)

        lens = x_seq.size(1) - padded
        x_seq = nn.utils.rnn.pack_padded_sequence(x_seq, lens.tolist(), batch_first=True)
        
        rnn_outputs = self.rnn_layers(x_seq)
        h = rnn_outputs[1][0] if self.rnn_type == "LSTM" else rnn_outputs[1]
        x_seq = torch.cat([h[-2], h[-1]], dim=-1) if self.rnn_bidirectional else h[-1]

        if self.batch_norm:
            x_seq = self.rnn_bn(x_seq)
        return x_seq


class MlpModule(NetFrame):
    def __init__(self, in_sz, mlp_config, batch_norm=True):
        """
        MLP module for processing statistic (fixed-size) data.

        in_sz (int): Input size.
        mlp_config (dict): Configuration dictionary for the MLP module.
        batch_norm (bool, optional): Whether to use batch normalization, default is True.
        """
        super().__init__()
        self.batch_norm = batch_norm
        self.out_sz = mlp_config["out_sz"]

        seq = []
        for i in range(mlp_config["depth"]):
            layer_in = in_sz if i == 0 else mlp_config["feat_sz"]
            layer_out = self.out_sz if i == mlp_config["depth"] - 1 else mlp_config["feat_sz"]
            seq.append(nn.Linear(layer_in, layer_out))
            if self.batch_norm:
                seq.append(nn.BatchNorm1d(layer_out))
            if i < mlp_config["depth"] - 1:
                seq.append(nn.GELU())
        self.mlp_layers = nn.Sequential(*seq)

    def forward(self, x):
        """
        x (torch.Tensor): Input tensor with shape (n_batch, feat_sz).
        ---
        output (torch.Tensor): Output tensor with shape (n_batch, out_sz).
        """
        return self.mlp_layers(x)


class EncoderNet(NetFrame):
    def __init__(self, config):
        """
        Encoder network for processing both static and sequential data.

        config (dict): Configuration dictionary for the encoder network.
        """
        super().__init__()
        self.series_data = config["traj_sz"] > 0
        self.batch_norm = config["batch_norm"]

        # encoder module for static (fixed-size) data
        self.stat_encoder = MlpModule(config["stat_sz"], config["mlp"], self.batch_norm)
        
        # encoder module for trajectory data
        if self.series_data:
            self.traj_encoder_type = config["traj_encoder_type"]
            if self.traj_encoder_type == "transformer":
                self.traj_encoder = TransformerModule(config["traj_sz"], config["transformer"])
            elif self.traj_encoder_type == "conv_pool":
                self.traj_encoder = ConvPoolingModule(config["traj_sz"], config["conv1d"], self.batch_norm)
            elif self.traj_encoder_type == "conv_rnn":
                self.traj_encoder = ConvRnnModule(config["traj_sz"], config["conv1d"], config["rnn"], self.batch_norm)
            else:
                raise RuntimeError("traj_encoder_type should be among 'transformer', 'conv_pool', and 'conv_rnn'.")
            
        self.compute_total_params("encoder")

    def forward(self, x, x_seq=None, padded=None, mask=None):
        """
        x (torch.Tensor): Input tensor with shape (n_batch, stat_sz).
        x_seq (torch.Tensor, optional): Input sequence tensor with shape (n_batch, data_length, traj_sz).
        padded (optional): List of sizes of zero-paddings.
        mask (optional): Mask tensor.
        ---
        output (torch.Tensor): Output tensor with shape (n_batch, out_sz).
        """
        x = torch.FloatTensor(x).to(self.device)
        x = self.stat_encoder(x)

        if self.series_data:
            assert x_seq is not None
            x_seq = torch.FloatTensor(x_seq).to(self.device)
            x_seq = self.traj_encoder(x_seq, padded, mask)
            x = torch.cat([x_seq, x], dim=-1)

        return x
    
    def compute_output_sz(self):
        """
        Compute the total output size of the encoder modules.
        """
        if self.series_data:
            return self.stat_encoder.out_sz + self.traj_encoder.out_sz
        else:
            return self.stat_encoder.out_sz


class TrialAttentionNet(NetFrame):
    def __init__(self, in_sz, config):
        """
        Encoder network based on attention across multiple trials.

        in_sz (int): Input size.
        config (dict): Configuration dictionary for the encoder network.
        """
        super().__init__(config)
        num_latents = config["num_latents"]
        n_block = config["n_block"]
        latent_sz = config["query_sz"]
        out_sz = config["out_sz"]

        config["context_sz"] = in_sz

        self.latents = nn.Parameter(torch.randn(num_latents, latent_sz))
        self.blocks = nn.Sequential(*[AttentionBlock(config) for _ in range(n_block)])
        self.head = nn.Sequential(
            Reduce("b n d -> b d", "mean"),
            nn.LayerNorm(latent_sz),
            nn.Linear(latent_sz, out_sz)
        )
        self.compute_total_params("trial-attention")

    def forward(self, context):
        """
        context (torch.Tensor): Input tensor with shape (n_batch, n_trial, context_sz).
        ---
        output (torch.Tensor): Output tensor with shape (n_batch, out_sz).
        """
        if len(context.shape) < 3:
            context = context.unsqueeze(0)
        b = context.shape[0]
        x = repeat(self.latents, "n d -> b n d", b=b)
        for block in self.blocks:
            x = block(x, context)
        return self.head(x)


class TrialInvariantNet(NetFrame):
    def __init__(self, in_sz, config):
        """
        Encoder network based on permutation-invariant representations across multiple trials.
        Reference:
        https://github.com/stefanradev93/BayesFlow/blob/master/bayesflow/networks.py

        in_sz (int): Input size.
        config (dict): Configuration dictionary for the encoder network.
        """
        super().__init__()
        n_block = config["n_block"]
        config["in_sz"] = in_sz

        equiv_layers = [EquivariantModule(config) for _ in range(n_block)]
        self.equiv_seq = nn.Sequential(*equiv_layers)
        self.inv_module = InvariantModule(config)
        self.compute_total_params("trial-invariant")
    
    def forward(self, x):
        """
        x (torch.Tensor): Input tensor with shape (n_batch, n_trial, context_sz).
        ---
        output (torch.Tensor): Output tensor with shape (n_batch, out_sz).
        """
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        N = int(x.shape[1])

        out_equiv = self.equiv_seq(x)
        out_inv = self.inv_module(out_equiv)
        return out_inv