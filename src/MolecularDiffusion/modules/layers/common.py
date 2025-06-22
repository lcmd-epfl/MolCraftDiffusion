import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module consisting of:
      - A linear layer to expand the input dimension by 4x
      - GELU activation
      - A linear projection back to the original dimension
      - Dropout for regularization

    Args:
        config: An object with attributes:
            - n_embd (int): The embedding dimension of the input and output.
            - bias (bool): Whether to use bias in linear layers.
            - dropout (float): Dropout probability.
    """

    def __init__(self, config):
        """
        Initializes the MLP module with the given configuration.

        Args:
            config: An object with attributes n_embd, bias, and dropout.
        """
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (..., n_embd)

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SinusoidalPositionEmbedding(nn.Module):
    """
    Positional embedding based on sine and cosine functions, proposed in `Attention Is All You Need`_.

    .. _Attention Is All You Need:
       https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
       output_dim (int): output dimension
    """

    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        inverse_frequency = 1 / (
            10000 ** (torch.arange(0.0, output_dim, 2.0) / output_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency)

    def forward(self, input):
        """"""
        # input: [B, L, ...]
        positions = torch.arange(
            input.shape[1] - 1, -1, -1.0, dtype=input.dtype, device=input.device
        )
        sinusoidal_input = torch.outer(positions, self.inverse_frequency)
        position_embedding = torch.cat(
            [sinusoidal_input.sin(), sinusoidal_input.cos()], -1
        )
        return position_embedding


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = (
            2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
        )
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()

