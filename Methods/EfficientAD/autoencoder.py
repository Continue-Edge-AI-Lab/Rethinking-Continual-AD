import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Autoencoder(nn.Module):
    """
    Autoencoder to find logical & global anomalies,
    taken directly from Table 8 of the paper.

    Takes in an image of size (B, 3, 256, 256)
    and outputs a feature map of size (B, 384, 64, 64)
    """
    def __init__(self):
        """
        Initialization of the Autoencoder
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # EncConv1
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # EncConv2
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # EncConv3
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # EncConv4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # EncConv5
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # EncConv6
            nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)
        ) # (B, 64, 1, 1)

        # Decoder
        self.decoder = nn.Sequential(
            # DecConv1
            nn.UpsamplingBilinear2d(size=3),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # DecConv2
            nn.UpsamplingBilinear2d(size=8),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # DecConv3
            nn.UpsamplingBilinear2d(size=15),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # DecConv4
            nn.UpsamplingBilinear2d(size=32),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # DecConv5
            nn.UpsamplingBilinear2d(size=63),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # DecConv6
            nn.UpsamplingBilinear2d(size=127),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            # DecConv7
            # NOTE: Instead of doing bilinear downsampling here, why don't we use another convolution?
            nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # DecConv8
            nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=1)
        ) # (B, 384, 64, 64)

        # Get device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

        return

    def forward(self, x: torch.Tensor):
        """
        Takes in an image tensor and outputs a feature map tensor.
        :param x: a torch.tensor of size (B, 3, 256, 256)
        :return: a torch.tensor of size (B, 384, 64, 64)
        """
        latent = self.encoder(x) # (B, 64, 1, 1)
        recon = self.decoder(latent) # (B, 384, 64, 64)

        return recon