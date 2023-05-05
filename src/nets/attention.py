"""
Reference:
https://github.com/lucidrains/perceiver-pytorch
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat


class Attention(nn.Module):
    """
    Vanila multi-head (cross or self) attention layer
    """
    def __init__(self, config, self_attn=False):
        super().__init__()
        # Extract configurations
        query_sz = config["query_sz"]
        context_sz = config["query_sz"] if self_attn else config["context_sz"]
        head_sz = config["head_sz"]
        n_head = config["n_head"]
        feat_sz = head_sz * n_head

        # Initialize variables
        self.self_attn = self_attn
        self.scale = head_sz ** -0.5
        self.n_head = n_head

        # Initialize layers
        self.to_q = nn.Linear(query_sz, feat_sz, bias=False)
        self.to_kv = nn.Linear(context_sz, feat_sz * 2, bias=False)
        self.dropout = nn.Dropout(config["attn_dropout"])
        self.to_out = nn.Linear(feat_sz, query_sz)

    def forward(self, x, context=None, mask=None):
        """
        x (torch.Tensor): Input query tensor, shape (batch_size, num_queries, query_sz).
        context (torch.Tensor): Input context tensor, shape (batch_size, num_context_points, context_sz).
        mask (torch.BoolTensor, optional): Attention mask, shape (batch_size, num_queries, num_context_points).
        ---
        outputs(torch.Tensor): Output tensor, shape (batch_size, num_queries, query_sz).
        """
        h = self.n_head
        q = self.to_q(x)
        if context is None:
            assert self.self_attn
            context = x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Rearrange tensors for multi-head attention
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # Calculate attention scores
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        # Apply mask if provided
        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)

        # Softmax and dropout
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # Calculate output using attention scores
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class AttentionBlock(nn.Module):
    """
    # Perceiver block where cross- & self-attention is repeated
    """
    def __init__(self, config):
        super().__init__()
        # Extract configurations
        query_sz = config["query_sz"]
        context_sz = config["context_sz"]

       # Initialize layers for cross-attention
        self.cross_ln_q = nn.LayerNorm(query_sz)
        self.cross_ln_kv = nn.LayerNorm(context_sz)
        self.cross_attn = Attention(config)
        self.cross_ln = nn.LayerNorm(query_sz)
        self.cross_mlp = nn.Sequential(
            nn.Linear(query_sz, 4 * query_sz),
            nn.GELU(),
            nn.Linear(4 * query_sz, query_sz),
            nn.Dropout(config["res_dropout"])
        )

        # Initialize layers for self-attention
        self.self_ln_qkv = nn.LayerNorm(query_sz)
        self.self_attn = Attention(config, self_attn=True)
        self.self_ln = nn.LayerNorm(query_sz)
        self.self_mlp = nn.Sequential(
            nn.Linear(query_sz, 4 * query_sz),
            nn.GELU(),
            nn.Linear(4 * query_sz, query_sz),
            nn.Dropout(config["res_dropout"])
        )

    def forward(self, x, context, mask=None):
        """
        x (torch.Tensor): Input query tensor, shape (batch_size, num_queries, query_sz).
        context (torch.Tensor): Input context tensor, shape (batch_size, num_context_points, context_sz).
        mask (torch.BoolTensor, optional): Attention mask, shape (batch_size, num_queries, num_context_points).
        ---
        outputs(torch.Tensor): Output tensor, shape (batch_size, num_queries, query_sz).
        """

        # Apply cross-attention followed by MLP
        x = x + self.cross_attn(self.cross_ln_q(x), self.cross_ln_kv(context), mask=mask)
        x = x + self.cross_mlp(self.cross_ln(x))

        # Apply self-attention followed by MLPs
        x = x + self.self_attn(self.self_ln_qkv(x))
        x = x + self.self_mlp(self.self_ln(x))
        return x