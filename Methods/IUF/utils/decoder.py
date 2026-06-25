"""
Class to create the Decoder network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # Library for cleaner tensor reshaping operations
from ..ViT import ViT

class Decoder(ViT):
    """

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.deconv = nn.ConvTranspose2d(
            in_channels=self.embedding_dim,  # E = 64
            out_channels=3,                  # RGB channels
            kernel_size=self.patch_size,     # 16
            stride=self.patch_size           # 16
        ).to(self.device)

        return

    def forward(self, z):
        # Takes in image input of size (B, 3, 224, 224)
        # Returns either the features list or the final output

        out = rearrange(super().forward(z=z), 'B (Ph Pw) E -> B E Ph Pw', Ph=self.patch_dim)
        # (B, E, P, P)
        # where
        # - E = embedding_dimension = 64 (default)
        # - P = patch_dim = 14

        x_recon = self.deconv(out)
        # (B, 3, 224, 224)

        return x_recon

