"""
Class to create the Encoder network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # Library for cleaner tensor reshaping operations
from ..ViT import ViT

class Encoder(ViT):
    """

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, oasa_features):
        # Takes in image input of size (B, 3, 224, 224)
        # As well as the oasa_features produced from the discriminator
        # Returns either the features list or the final output

        # Producing latent encoding, z
        out = super().forward(x, oasa_features=oasa_features)
        # out, and all tensors in features, have shape (B, L, E)
        # where
        # - L = sequence length = num_patches = patch_size**2 = 196 (default)
        # - E = embedding_dimension = 64 (default)

        # Re-arrange back into patch format, (B x E x P_d x P_d), for aggregation
        z = rearrange(out, 'B (Ph Pw) E -> B E Ph Pw', Ph=self.patch_dim)

        # Aggregates each patch by taking the mean of that patch, across all channels
        m_hat = torch.mean(z, dim=(2,3))
        # Returns m_hat of size (B, E), where the E = embedding dimension
        # is treated as the number of channels

        # Take SVD — computed on CPU to avoid cuSOLVER dependency on Jetson
        # m_hat is small (B x E), so CPU overhead is negligible
        u, s, v = torch.linalg.svd(m_hat.cpu())
        u, s, v = u.to(m_hat.device), s.to(m_hat.device), v.to(m_hat.device)
        # u = (B, B), basis for batch space (not used)
        # s = (B) = singular values
        # v = (E, E), basis for channel space

        return out, u, s, v
