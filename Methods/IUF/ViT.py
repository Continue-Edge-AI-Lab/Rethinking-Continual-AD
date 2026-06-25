"""
Base ViT class for IUF, used to create the
discriminator, encoder, and decoder.

Based on author's implementation (original_code/reconstructions/ViT.py),
and it should be noted that code experiments use different models for the encoder
(see original_code/experiments config.yaml files),
but the paper explicitly shows that ViT's are used for all 3 component models:
- Discriminator
- Encoder
- Decoder

TODO:
    - Add gradient weight update (IUF section 4.3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # Library for cleaner tensor reshaping operations

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism that operates on 2D feature maps rather than
    tokenized sequences as in standard ViT implementations. Takes in an input
    of size (B x embed_dim x H x W), where the default embed_dim = 64.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        # Ensure dimensions are compatible with multi-head mechanism
        assert embed_dim % num_heads == 0, "embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads  # Number of attention heads
        self.embed_dim = embed_dim // num_heads

        # Linear projections for query, key, value
        # Maintains input shape, but gives each pixel its own transformation
        # the Linear MLP takes a vector of size (*, H_in), where * means any number of dimensions
        # and outputs a vector of size (*, H_out)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, oasa_features=None):
        # x,oasa_features,q,k,v shape: (B x L x E)
        # Linear projections
        q = self.query(x)
        # Use Hadamard product with OASA features, if not None
        if oasa_features is not None:
            q = q * oasa_features
        k = self.key(x)
        v = self.value(x)

        # Reshape for each head, where
        # B = batch_size, L = sequence_length, n = num_heads, d = head_dim
        q = rearrange(q, 'B L (n d) -> n B L d', n=self.num_heads)
        k = rearrange(k, 'B L (n d) -> n B d L', n=self.num_heads)
        v = rearrange(v, 'B L (n d) -> n B L d', n=self.num_heads)

        # returns matrix of size (N, B, L, L), where
        # each row in each LxL matrix is the vector of attention weights for embedding l in L
        attn_weights = self.softmax(torch.matmul(q, k) / (self.embed_dim ** 0.5))

        # returns matrix of size (N, B, L, d), where
        # each row in each LxL matrix is the weighted sum of the values for embedding l in L
        attn_scores = torch.matmul(attn_weights, v)

        # Concatenate heads back together
        output = rearrange(attn_scores, 'n B L d -> B L (n d)')  # (B, L, E)

        # Apply output projection (common in transformers)
        output = self.out_proj(output)

        return output

class ViTBlock(nn.Module):
    """
    A single transformer encoder block with self-attention and MLP.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.MHSA = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Missing this:
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # Expansion
            nn.GELU(),                            # Activation
            nn.Linear(4 * embed_dim, embed_dim)   # Back to original size
        )

    def forward(self, x, oasa_features=None):
        """
        Takes in a tensor of shape [B x L x embed_dim]
        and outputs a tensor of the same shape.
        """
        
        # First sub-block: Normalization + Self-Attention + Residual
        out = self.MHSA(self.norm1(x), oasa_features)  # Apply norm, then Multi-head Self Attention
        x = x + out                             # Residual connection

        # Second sub-block: Normalization + MLP + Residual
        x = x + self.mlp(self.norm2(x))         # Apply norm, MLP, then residual
        return x


# noinspection DuplicatedCode
class ViT(nn.Module):
    """
    Modified Vision Transformer, where the last layer output is returned, plus intermediary outputs if necessary.
    The idea is that for each of the 3 ViT's in IUF, they can add their own classification head.

    All params are the same as original IUF paper, unless otherwise specified.
    Recall, the embed_dim is split across all heads, so embed_dim % num_heads must be 0.

    Notes:
        - Added positional encoding to the tokens
        - Changed BatchNorm to LayerNorm and ReLU to GELU, in accordance with the ViT paper
        - I noticed that in their Multi-head attention, the authors only work on row-wise attention to simplify
        their computations, so I am going to operate on patches, as the matrix multiplication is too large.
    """
    def __init__(self,
                 device,
                 in_channels=3,
                 patch_size=16, # 224 should be divisible by patch_size
                 embed_dim=64,
                 num_heads=4, # Should be able to take the sqrt easily
                 num_layers=4
                 ):

        super().__init__()
        # Check inputs
        assert 224 % patch_size == 0, "Image size must be divisible by patch size"

        self.patch_dim = 224 // patch_size
        self.num_patches = (self.patch_dim)** 2 # 196

        self.patch_size = patch_size
        self.embedding_dim = embed_dim

        # Initial convolutional embedding
        self.conv1 = nn.Conv2d(in_channels,
                               embed_dim,
                               kernel_size=patch_size,
                               stride=patch_size)
        # Add learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()

        # Stack of ViT blocks
        self.vit_blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads)
            for _ in range(num_layers)  # 4 transformer blocks
        ])
        self.num_layers = num_layers
        self.to(device)
        self.device = device

    def forward(self,
                x=None,
                z=None,
                return_features=False,
                oasa_features=None):
        """
        Takes in an image tensor of shape [B x 3 x 224 x 224] or [B x L x E].
        ** If return_features==True, oasa_features must be None
        ** If oasa_features is not none, return features must be False.
        Args:
            x: image tensor of shape [B x 3 x 224 x 224]
            z: ViT latent space outputs of shape [B x L x embed_dim]
            return_features: Bool, whether to return a list of the intermediate features.
                                If false, returns only the final layer output. Each layer's output
                                is a tensor of shape [B x sequence_length x embed_dim].
                                Only used for discriminator
            oasa_features: a list of detached tensors from the discriminator

        Returns:

        """
        # Expects images to be of shape (B, 3, 224, 224)
        # Outputs either the last or all ViT layers, which will have shape (B, L, E), where
        # B = batch_size,
        # L = sequence_length = (224/patch_size)**2 (default L = 14x14 = 196),
        # E = embed_dim (default is 64)

        if z is None:
            assert x is not None, "x and z cannot both be none"
            out = rearrange(self.conv1(x), 'B E PH PW -> B (PH PW) E') # (B, PxP = L, E), so each patch has its own embedding
        else:
            out = z
        # Add positional embedding
        out = out + self.pos_embedding
        # Normalize
        out = self.ln1(out)
        out = self.gelu(out)

        # Store intermediate outputs for OASA features
        # return_feature_outputs adds outputs of all layers
        if return_features:
            features = []
        # Pass through transformer blocks
        for l in range(self.num_layers):
            vit_block = self.vit_blocks[l]

            if (oasa_features is not None) and (l != self.num_layers - 1):
                out = vit_block(out, oasa_features[l])  # [batch, sequence_length, embed_dim]
            else:
                out = vit_block(out)

            if return_features:
                features.append(out.detach())  # Store intermediate representations
        # layer output = (B, L, E), where
        # B = batch_size, L = sequence_length, E = embed_dim

        # Return output
        if return_features:
            return features, out
        else:
            return out
