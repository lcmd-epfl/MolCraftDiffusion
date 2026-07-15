
import torch
import torch.nn as nn
import math

class GaussianFeatureEmbedding(nn.Module):
    def __init__(self, num_channels, num_basis=20, start=0.0, stop=10.0, trainable=False):
        super().__init__()
        self.num_channels = num_channels
        self.num_basis = num_basis
        self.trainable = trainable
        
        offset = torch.linspace(start, stop, num_basis)
        self.register_buffer("offset", offset)
        
        # Width of the Gaussian 
        # width = (stop - start) / num_basis # This might be too narrow if simple spacing
        # Suggest width = distance between centers
        width = (stop - start) / (num_basis - 1) if num_basis > 1 else 1.0
        self.register_buffer("width", torch.tensor(width))

        if trainable:
            self.offset = nn.Parameter(self.offset)
            self.width = nn.Parameter(self.width)

    def forward(self, x):
        # x: [Batch, N, num_channels]
        # output: [Batch, N, num_channels * num_basis]
        
        # Expand dims
        # x: [B, N, C, 1]
        x_expanded = x.unsqueeze(-1)
        
        # offset: [1, 1, 1, Basis]
        offset = self.offset.view(1, 1, 1, -1)
        
        # coeff: [B, N, C, Basis]
        coeff = -0.5 * torch.pow(x_expanded - offset, 2) / torch.pow(self.width, 2)
        g = torch.exp(coeff)
        
        # Flatten last two dims
        # [B, N, C * Basis]
        return g.reshape(x.size(0), x.size(1), -1)

