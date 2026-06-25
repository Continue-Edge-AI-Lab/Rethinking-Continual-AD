"""
Base ViT class for UCAD

Used to create the slightly custom frozen ViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from einops import rearrange  # Library for cleaner tensor reshaping operations

class ViT(nn.Module):
    def __init__(self,
                 output_layer=5,
                 device='cpu'
                 ):
        super().__init__()

        # Our pre-trained ViT
        base_vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Initial Convolutional Projection
        self.conv_proj = base_vit.conv_proj

        # Get class token
        self.class_token = base_vit.class_token # (1, 1, 768)

        # Get positional encoding vector
        # (1, 196, 768)
        self.pos_vec = base_vit.encoder.pos_embedding[:, 0:196, :]


        # vit modules will hold all modules until we add them into a sequential for
        self.vit_encoder = nn.ModuleList()
        for i in range(output_layer):
            self.vit_encoder.append(base_vit.encoder.layers[i])

        # Get device
        self.to(device)
        self.device = device

        return

    def forward(self, x, prompt_vec=None):
        x =x.to(self.device)

        # Get initial projection
        # (B, 768, 14, 14)
        out = self.conv_proj(x)
        # Re-arrange
        # (B, 196, 768)
        out = rearrange(out, 'b c h w -> b (h w) c')
        # Add positional encoding
        # (B, 196, 768)
        out += (self.pos_vec.to(self.device))

        # Pass encoded tokens into encoder MHSA layers
        for j in range(len(self.vit_encoder)):
            # Add prompt vector first, if it was included in input
            if prompt_vec is not None:
                out += prompt_vec[j].expand(196, -1)
            out = self.vit_encoder[j](out)

        return out