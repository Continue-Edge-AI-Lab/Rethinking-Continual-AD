import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ..PatchCore.resnet import ResNet

class PDN(nn.Module):
    """
    Patch Description Network, taken directly from Tables 6 & 7 of the paper.

    Takes in an image of size (B, 3, 256, 256)
    and outputs a feature map of size (B, C, 64, 64),
    where
        - B = batch size
        - C = number of channels, 384 for teacher and 768 for student
    """
    def __init__(self,
                 size: str = 's',
                 student: bool = False,):
        """
        Initialization of the PDN.

        :param size: (str), either 's' for small or 'm' for medium
        """
        super().__init__()

        num_output_channels = 768 if student else 384

        self.size = size
        if size == 's':
            self.layers = nn.Sequential(
                # Conv-1
                nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
                # Conv-2
                nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
                # Conv-3
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # Conv-4
                nn.Conv2d(256, num_output_channels, kernel_size=4, stride=1, padding=0)
            )
        elif size == 'm':
            self.layers = nn.Sequential(
                # Conv-1
                nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
                # Conv-2
                nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
                # Conv-3
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                # Conv-4
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # Conv-5
                nn.Conv2d(512, 384, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                # Conv-6
                nn.Conv2d(384, num_output_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise ValueError('Size must be either "s" or "m"')

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
        return self.layers(x)