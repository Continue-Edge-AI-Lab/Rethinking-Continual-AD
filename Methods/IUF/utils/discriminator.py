"""
Class to create the discriminator network
"""

import torch.nn as nn
from einops import rearrange
from ..ViT import ViT

class Discriminator(ViT):
    """

    """
    def __init__(self, output_size=15, **kwargs):
        super().__init__(**kwargs)

        input_dim = (self.patch_dim ** 2) * self.embedding_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim*4),
            nn.GELU(),
            nn.Linear(input_dim*4, output_size)
        ).to(self.device)

    def forward(self, x, return_features=True):
        # Takes in image input of size (B, 3, 224, 224)
        # Returns either the features list or the final output

        features, out = super().forward(x, return_features=return_features)
        # out, and all tensors in features, have shape (B, L, E)
        # where
        # - L = sequence length = num_patches
        # - E = embedding_dimension = 64 (default)

        # Re-arrange for MLP
        out = rearrange(out, 'B L E -> B (L E)')
        out = self.mlp(out)
        # Outputs (B, num_classes)

        return features, out
