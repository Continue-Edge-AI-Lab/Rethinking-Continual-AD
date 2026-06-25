# __init__.py is called when imported by another Python file, where this directory is a python package, consisting modules of .py files

# Setting __all__ tells Python which modules (.py files) to import when importing this package
# __all__ = ['dne']

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torcheval.metrics.functional as tef


# =============================================================================
# Pixel-level metric utilities (added for ECCV 2026 rebuttal — pixel AUROC).
#
# Context: each dataset class (datasets/mvtec.py, mvtec_loco.py, mtd.py) applies
# `transforms.Normalize(mean=ImageNet, std=ImageNet)` to BOTH the input image
# AND the ground-truth mask. This shifts binary 0/255 masks into approximately
# [-2.12, +2.64] across channels — no longer binary. We denormalize here rather
# than touching the dataset class to avoid breaking other code paths.
# =============================================================================

# ImageNet normalization constants used by every dataset's default transform.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def recover_binary_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Recover a binary 0/1 ground-truth mask from a tensor that was passed through
    the dataset's default transform pipeline (Resize -> ToDtype(scale=True) ->
    Normalize(ImageNet stats)).

    The original GT masks contain pixel values 0 (normal) or 255 (anomalous);
    after `scale=True` they are 0.0 or 1.0; after Normalize they are negative
    in every channel for original-zero pixels and positive in every channel for
    original-one pixels.

    We recover binary by denormalizing back to [0, 1] then thresholding at 0.5.
    Any channel above 0.5 -> mark pixel as anomaly.

    Args:
        mask: tensor of shape (H, W), (C, H, W), or (N, C, H, W). If shape is
              (H, W) or (N, H, W) we assume it is already binary or single-channel
              and threshold at 0.5 directly.

    Returns:
        binary mask tensor (int) of shape (H, W) (if input was 2D or 3D) or
        (N, H, W) (if input was 4D).
    """
    if mask.dim() == 2:
        # (H, W) — assume already binary or unnormalized, threshold at 0.5
        return (mask > 0.5).int()

    if mask.dim() == 3:
        # Could be (C, H, W) or (N, H, W). Disambiguate by first-dim size.
        if mask.shape[0] == 3:
            # (C=3, H, W) — denormalize then OR across channels
            mean = _IMAGENET_MEAN.to(mask.device)
            std = _IMAGENET_STD.to(mask.device)
            denorm = mask * std + mean
            return (denorm > 0.5).any(dim=0).int()
        elif mask.shape[0] == 1:
            # (C=1, H, W) — single channel after normalization
            mean = _IMAGENET_MEAN[0:1].to(mask.device)
            std = _IMAGENET_STD[0:1].to(mask.device)
            denorm = mask * std + mean
            return (denorm.squeeze(0) > 0.5).int()
        else:
            # (N, H, W) — already 2D per image
            return (mask > 0.5).int()

    if mask.dim() == 4:
        # (N, C, H, W) — denormalize, OR across channel dim
        mean = _IMAGENET_MEAN.unsqueeze(0).to(mask.device)
        std = _IMAGENET_STD.unsqueeze(0).to(mask.device)
        denorm = mask * std + mean
        return (denorm > 0.5).any(dim=1).int()

    raise ValueError(f"recover_binary_mask: unexpected mask shape {tuple(mask.shape)}")


def compute_pixel_auroc(pixel_scores: torch.Tensor,
                        gt_masks_binary: torch.Tensor) -> float:
    """
    Compute pooled pixel-level AUROC across all images in a test set.

    Standard MVTec-AD evaluation pools pixels from ALL test images (normal +
    anomalous) into a single AUROC computation. Normal images contribute false-
    positive opportunities; anomalous images contribute true-positive opportunities.

    Args:
        pixel_scores: (N, H, W) anomaly scores at the same spatial resolution
                      as gt_masks_binary. Caller is responsible for any required
                      upsampling (e.g., 14x14 patch scores -> 224x224).
        gt_masks_binary: (N, H, W) binary 0/1 ground truth.

    Returns:
        float pixel AUROC, or NaN if a degenerate case (all pixels normal /
        all pixels anomalous, no test images, etc.).
    """
    if pixel_scores.shape != gt_masks_binary.shape:
        raise ValueError(
            f"compute_pixel_auroc: shape mismatch {tuple(pixel_scores.shape)} vs "
            f"{tuple(gt_masks_binary.shape)} — caller must upsample first."
        )

    if pixel_scores.numel() == 0:
        return float('nan')

    scores_flat = pixel_scores.flatten().float().cpu()
    labels_flat = gt_masks_binary.flatten().int().cpu()

    n_pos = int(labels_flat.sum().item())
    n_neg = int(labels_flat.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    return tef.binary_auroc(scores_flat, labels_flat).item()


def upsample_anomaly_map(scores: torch.Tensor, target_size: int = 224) -> torch.Tensor:
    """
    Bilinearly upsample a per-image anomaly map to (target_size, target_size).

    Args:
        scores: (N, h, w) low-resolution anomaly maps (e.g., 14x14, 28x28).
        target_size: int, output spatial size (default 224 to match GT mask).

    Returns:
        (N, target_size, target_size) upsampled scores, float32.
    """
    if scores.dim() != 3:
        raise ValueError(
            f"upsample_anomaly_map expects (N, h, w), got {tuple(scores.shape)}"
        )
    # (N, h, w) -> (N, 1, h, w) for F.interpolate, which expects (N, C, H, W)
    s = scores.float().unsqueeze(1)
    s = F.interpolate(s, size=(target_size, target_size),
                      mode='bilinear', align_corners=False)
    return s.squeeze(1)


def stack_gt_masks(gt_masks_list: list, target_size: int = 224) -> torch.Tensor:
    """
    Convert a list of per-image GT masks (any of: None, (1,H,W), (3,H,W),
    (H,W)) into a stacked binary tensor of shape (N, target_size, target_size).

    None entries -> all-zero masks. Mixed-channel masks are denormalized via
    recover_binary_mask. Differently-sized masks are bilinearly resized to
    (target_size, target_size) BEFORE binarization to avoid losing the binary
    structure when interpolating already-binary tensors.

    Args:
        gt_masks_list: list of tensors-or-None as returned by per-sample
                       collation (each entry came from dataset['ground_truth_mask']).
        target_size: spatial resolution to stack to. Match this to the
                     resolution of the model's pixel anomaly map.

    Returns:
        (N, target_size, target_size) int binary tensor.
    """
    out = []
    for m in gt_masks_list:
        if m is None:
            out.append(torch.zeros(target_size, target_size, dtype=torch.int))
            continue

        # Resize first if needed (in normalized space — float)
        # Add batch+channel dims for F.interpolate
        m_t = m.float()
        if m_t.dim() == 2:
            m_t = m_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            had_2d = True
        elif m_t.dim() == 3:
            m_t = m_t.unsqueeze(0)  # (1, C, H, W)
            had_2d = False
        else:
            raise ValueError(f"Unexpected per-sample mask shape {tuple(m.shape)}")

        if m_t.shape[-1] != target_size or m_t.shape[-2] != target_size:
            m_t = F.interpolate(m_t, size=(target_size, target_size),
                                mode='bilinear', align_corners=False)

        # Strip the batch dim, then binarize via recover_binary_mask
        m_t = m_t.squeeze(0)  # (C, H, W) or (1, H, W) for the originally-2D case
        if had_2d:
            m_t = m_t.squeeze(0)  # (H, W)

        binary = recover_binary_mask(m_t)
        out.append(binary)

    return torch.stack(out, dim=0).int()

class BaseAnomalyDetector(nn.Module):
    """Base class for anomaly detection models in continual learning scenarios.

    This class provides common functionality for:
    - Setting up device management
    - Base model construction
    - Training/inference patterns
    - Memory management for continual learning
    """
    def __init__(self):
        """
        Initialize the base anomaly detector.
        """
        super().__init__()

        # Set up device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using {self.device} device")

        # Move to device
        self.to(self.device)

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def embed(self, x):
        """Extract embeddings from the input tensor.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Embeddings tensor
        """
        raise NotImplementedError("Subclasses must implement embed method")

    def train_one_epoch(self, dataloader, optimizer, criterion, task_num, **kwargs):
        """Train the model for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader for training data
            optimizer (torch.optim.Optimizer): Optimizer
            criterion (callable): Loss function
            task_num (int): Current task number
            **kwargs: Additional keyword arguments

        Returns:
            float: Total Loss for that epoch
        """
        raise NotImplementedError("Subclasses must implement train_one_epoch method")

    def eval_one_epoch(self, dataloader, criterion, task_num, **kwargs):
        """
        Evaluate the model for one epoch of a testing set

        Args:
            dataloader:
            criterion:
            task_num:
            **kwargs:

        Returns:
            float: Total Loss for that epoch
        """
        raise NotImplementedError("Subclasses must implement eval_one_epoch method")

    def calc_results(self, dataset, exp, metrics):
        """
        Method used to calculate results for a given model and save that data as CSV
        Args:
            dataset (str): 'MTD' or 'MVTEC'
            exp (str): 'unsupervised' or 'supervised'
        Returns:

        """

        raise NotImplementedError("Subclasses must implement calc_results method")

    def predict(self, x):
        """Make prediction for input.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Prediction
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def save(self, path):
        """
        Saves the model to disk.
        Args:
            path: path to save the model to.
        """
        torch.save(self.state_dict(), path)
        return

    def load(self, path):
        """
        Loads the model from disk.
        Args:
            path: path to load the model from.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        return



