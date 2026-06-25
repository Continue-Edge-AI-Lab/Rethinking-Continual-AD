"""

"""
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    """
    Modified ResNet Backbone for SSD Feature Extraction

    This class adapts standard ResNet architectures for use as SSD backbones by:
    1. Removing the final classification layers
    2. Modifying stride in the last residual block to maintain spatial resolution
    3. Defining output channel dimensions for each feature pyramid level

    The backbone extracts features that will be fed into SSD detection heads.
    """

    def __init__(self, backbone='resnet18',
                 backbone_path=None,
                 weights="IMAGENET1K_V1",
                 device=torch.device('cpu'),
                 patchcore=False):
        """
        Initialize ResNet backbone for SSD

        Args:
            backbone (str): Which ResNet variant to use ('resnet18', 'resnet34', 'resnet50', etc.)
            backbone_path (str): Path to pre-trained weights file (optional)
            weights (str): Torchvision weights to use if backbone_path not provided
        """
        super().__init__()

        self.backbone_name = backbone
        self.device = device

        # Define output channel configurations for different ResNet variants
        # These channels correspond to feature maps at different pyramid levels
        if backbone == 'resnet18':
            backbone = resnet18(weights=None if backbone_path else weights)
            # For ResNet18: [level1, level2, level3, level4, level5, level6]
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(weights=None if backbone_path else weights)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(weights=None if backbone_path else weights)
            # ResNet50 has more channels due to bottleneck architecture
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet152':
            backbone = resnet152(weights=None if backbone_path else weights)
            self.out_channels = [1024, 512, 512, 256, 256, 256]

        self.to(self.device)
        backbone.to(self.device)
        self.patchcore = patchcore # Used to define which layers to output

        # Load custom pre-trained weights if provided
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        # Extract only the convolutional layers (remove FC layers and avgpool)
        # children()[:6] gives us: conv1, bn1, relu, maxpool, layer1, layer2
        # This corresponds to the first 2 residual blocks of ResNet
        backbone_children = list(backbone.children())
        self.feature_extractor = nn.Sequential(*backbone_children[:6]).to(self.device)
        # We separately extract the final layer
        self.final_layer = backbone_children[6]

        # if patchcore, we want to freeze grad computation
        if patchcore:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.final_layer.parameters():
                param.requires_grad = False
        # If not, then it's for SSD and we want to adjust our output
        else:
            # Modify the first block of layer3 (conv4_block1) to reduce stride
            # This keeps spatial resolution higher for better small object detection
            conv4_block1 = self.final_layer[0]  # First block of layer3

            # Change stride from (2,2) to (1,1) to maintain spatial resolution
            conv4_block1.conv1.stride = (1, 1)
            conv4_block1.conv2.stride = (1, 1)
            # Also modify the downsample layer if it exists
            conv4_block1.downsample[0].stride = (1, 1)

        self.to(self.device)
        return



    def forward(self, x):
        """
        Forward pass through ResNet backbone

        Args:
            x (torch.Tensor): Input image tensor [batch_size, 3, height, width]

        Returns:
            if patchcore, then we will return two feature maps:
            torch.Tensor: Two feature maps from ResNet backbone:
                f1 - [B, 128, 38, 38] - feature map for level1
                f2 - [B, 256, 19, 19] - feature map for level2
            if not patchcore, then we are using SSD and only return f2:
                f2 - [B, 256, 38, 38] - feature map for level2
        """
        f1 = self.feature_extractor(x)
        f2 = self.final_layer(f1)
        if self.patchcore:
            return f1, f2
        else:
            return f2