import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader
import torcheval.metrics.functional as tef
from einops import rearrange
from .resnet import ResNet
import numpy as np
import os
import pandas as pd
from Methods import (
    BaseAnomalyDetector,
    compute_pixel_auroc,
    upsample_anomaly_map,
    stack_gt_masks,
)

class Patchcore_Model(BaseAnomalyDetector):
    """
    Complete PatchCore model

    Algorithm Notes:

        ViT
        - Added positional encoding to the tokens
        - Changed BatchNorm to LayerNorm and ReLU to GELU, in accordance with the ViT paper
        - I noticed that in their Multi-head attention, the authors only work on row-wise attention to simplify their computations, so I am going to operate on patches, as the multiplication is too large.
        - Batch size needs to maybe be larger than the embedding dimension?
    """
    def __init__(self,
                 backbone: str ='resnet50', # can also be 'resnet34', 'resnet50', or 'efficientnet_b0'
                 coreset_pct: float = 0.05
                 ):

        ### Our modules

        # Create memory bank
        self.memory_bank = None
        self.coreset_pct = coreset_pct

        # Get device and move all modules there
        super().__init__()

        # Get backbone, which also has all params have requires_grad=False
        if 'resnet' in backbone:
            self.backbone = ResNet(backbone=backbone, device=self.device, patchcore=True)
            if backbone == 'resnet18':
                self.embed_dim = 100
                linear_features = 384
            elif backbone == 'resnet34':
                pass
            elif backbone == 'resnet50':
                self.embed_dim = 512
                linear_features = 1536

        # Our average pooling layers
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1).to(self.device)

        # Bilinear upsampling
        self.upsample_fm2 = nn.UpsamplingBilinear2d(size=28).to(self.device)

        # Random linear projection
        self.random_projection = nn.Linear(in_features=linear_features,
                                           out_features=self.embed_dim).to(self.device)



        return

    def forward(self, x: torch.Tensor):
        """
        Passes batched images and gets final features for inference or training.
        fm = feature map
        :param x: image of size (B, C, 300, 300)
        :return: tensor of features (B, 28, 28, embed_dim)
        """
        # 1) Extract backbone features
        # x =
        fm1, fm2 = self.backbone(x)
        # output has 2 feature map with size
        # - (B, 128, 28, 28)
        # - (B, 256, 14, 14)

        # 2) Apply adaptive average pooling
        fm1 = self.avg_pool(fm1)
        fm2 = self.avg_pool(fm2)

        # 3) Bilinearly upsample 2nd feature map
        fm2 = self.upsample_fm2(fm2)

        # 4) Aggregate features into one map
        fm = torch.cat((fm1, fm2), dim=1)

        # 5) Downsample each feature vector using random linear projection
        fm = rearrange(fm, 'b c h w -> b h w c')
        fm = self.random_projection(fm)

        return fm

    def _coreset_sampling(self):
        """
        Applies coreset sampling to the memory bank.
        Therefore, the memory bank must already be created.
        It will turn the memory bank from size M => size M*coreset_pct
        """
        # Memory bank size and corresponding coreset size
        m_size = self.memory_bank.shape[0]
        coreset_size = max(500, int(m_size * self.coreset_pct))
        # Chosen coreset indices
        coreset_idx = torch.zeros(m_size).to(self.device)
        # randomly choose first index and calculate max distance
        random_idx = torch.randint(low=0, high=m_size, size=(1,1)).item()
        coreset_idx[random_idx] = 1
        chosen_feature = self.memory_bank[random_idx]
        min_dist = F.pairwise_distance(self.memory_bank, chosen_feature)
        while coreset_idx.sum() < coreset_size:
            # Get all non_chosen indices and zero out any already chosen indices' distance
            # So, it's not calculated for finding next max distance
            non_chosen_idx = abs(coreset_idx - 1).to(self.device)
            min_dist *= non_chosen_idx
            # Get new coreset point
            new_idx = min_dist.argmax()
            coreset_idx[new_idx] = 1
            chosen_feature = self.memory_bank[new_idx]
            # Calculate new minimum distance to new feature
            new_min_dist = F.pairwise_distance(self.memory_bank, chosen_feature)
            # If the new distance is less than the old distance, update min_dist
            min_dist = torch.min(min_dist, new_min_dist)

        # Get final coreset
        self.memory_bank = self.memory_bank[coreset_idx.bool()]

        return

    def predict(self, x: torch.Tensor, return_patch_scores: bool = False):
        """
        Vectorized PatchCore prediction using torch.cdist() instead of per-patch loops.

        :param x: batched image tensor of size (B, 3, 224, 224)
        :param return_patch_scores: if True, also returns the per-patch (B, 28, 28)
                                    anomaly map. Used for pixel-level metric computation.
                                    Default False keeps the legacy image-level-only API.
        :return: if return_patch_scores=False (default): image-level anomaly scores, a list of size B.
                 if return_patch_scores=True: tuple (img_ad_scores: list[float], patch_maps: (B, 28, 28) tensor)
        """
        with torch.no_grad():
            img_ad_scores = []
            patch_maps = []  # one (28, 28) tensor per image
            # (B, 28, 28, embed_dim)
            fm = self.forward(x)
            b, h, w, d = fm.shape
            memory = self.memory_bank  # (M, embed_dim)

            for k in range(b):
                # Flatten spatial dims: (28, 28, embed_dim) → (784, embed_dim)
                patches = fm[k].reshape(h * w, d)

                # Step 1: Find nearest memory bank entry for each patch — all at once
                # torch.cdist computes all pairwise L2 distances: (784, M)
                all_dists = torch.cdist(patches.to(self.device), memory)
                # Nearest memory bank entry per patch
                m_star_dists, m_star_indices = all_dists.min(dim=1)  # both (784,)
                m_stars = memory[m_star_indices]  # (784, embed_dim)

                # m_test_star is the patch with the maximum nearest-neighbor distance
                m_test_star_flat_idx = m_star_dists.argmax()
                row = m_test_star_flat_idx // w
                col = m_test_star_flat_idx % w

                # Step 2: Re-weighting (Equation 2 from PatchCore paper)
                # For each patch's nearest memory entry, find its k=3 neighbors in the memory bank
                m_star_to_memory_dists = torch.cdist(m_stars.to(self.device), memory)  # (784, M)
                _, nn_indices = torch.topk(m_star_to_memory_dists, k=3, largest=False, dim=1)  # (784, 3)

                # Gather the 3 nearest neighbors for each patch: (784, 3, embed_dim)
                nn_features = memory[nn_indices]

                # Distance from each test patch to its m_star's 3 neighbors: (784, 3)
                # patches: (784, 1, embed_dim), nn_features: (784, 3, embed_dim) → diff → norm → (784, 3)
                dists_to_neighbors = torch.norm(
                    patches.to(self.device).unsqueeze(1) - nn_features, dim=2
                )

                # Apply re-weighting: w = 1 - exp(s*) / sum(exp(d_neighbors))
                numerator = torch.exp(m_star_dists)           # (784,)
                denominator = torch.exp(dists_to_neighbors).sum(dim=1)  # (784,)
                weights = 1.0 - (numerator / denominator)     # (784,)
                patch_ad_scores_2d = (weights * m_star_dists).reshape(h, w)  # (h, w)

                # Image-level AD score = score of the most anomalous patch
                img_ad_scores.append(patch_ad_scores_2d[row, col].item())
                patch_maps.append(patch_ad_scores_2d.detach().cpu())

        if return_patch_scores:
            patch_maps = torch.stack(patch_maps, dim=0)  # (B, 28, 28)
            return img_ad_scores, patch_maps
        return img_ad_scores

    def train_one_epoch(self, dataloader: torch.utils.data.dataloader, **kwargs):
        """
        Given a dataloader of nominal data, fits the model to memory bank,
        then applies coreset sampling

        :param dataloader: dataloader of nominal data
        """
        with torch.no_grad():

            for batch_idx, data in enumerate(dataloader):

                # Get initial features
                imgs = data['image'].to(self.device) # (B, 3, 224, 224)
                fm = self.forward(imgs) # (B, 28, 28, embed_dim)
                fm = rearrange(fm, 'b h w c -> (b h w) c')

                # Add to memory bank
                if self.memory_bank is None:
                    self.memory_bank = fm.clone()
                else:
                    self.memory_bank = torch.cat((self.memory_bank, fm), dim=0)

            ### Apply coreset sampling
            self._coreset_sampling()
            # Turns memory bank to size (M_coreset, embed_dim)

        return 0

    def eval_one_epoch(self, dataloader, return_patch_scores: bool = False):
        """
        Evaluate model on one epoch, collecting raw model outputs for later analysis.

        Args:
            dataloader: DataLoader for the test/train dataset.
            return_patch_scores: if True, additionally returns per-image patch
                                 anomaly maps (N, 28, 28) and the raw GT mask
                                 list. Used by calc_results when the
                                 'pixel_auroc' metric is requested.

        Returns:
            If return_patch_scores=False (legacy):
                (ad_scores: tensor (N,), gt_masks: tensor, labels: tensor (N,))
            If return_patch_scores=True:
                (ad_scores, gt_masks, labels, patch_maps: (N, 28, 28),
                 gt_masks_raw: list[Tensor])

        Note: the legacy `gt_masks` return uses `torch.cat(..., dim=0)` over
        per-image (3, H, W) tensors, which produces shape (3N, H, W) — a known
        idiosyncrasy that has not mattered because gt_masks was never consumed
        downstream. We preserve that exact return shape for backward compat.
        Pixel-AUROC uses the raw list (gt_masks_raw) instead.
        """
        self.eval()

        img_ad_scores = []
        gt_masks_list = []          # raw per-image masks (or zero-filled stand-ins)
        labels = []
        patch_maps_list = []        # populated only if return_patch_scores=True

        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch_idx, data in enumerate(dataloader):

                # Get ad scores from the predict() method
                imgs = data['image'].to(self.device) # (B, 3, 224, 224)
                # These distances represent the distance from each image patch
                # to the nearest vector in the coreset memory bank,
                # and these would be the m_star variables in
                if return_patch_scores:
                    batch_ad_scores, batch_patch_maps = self.predict(imgs, return_patch_scores=True)
                    patch_maps_list.append(batch_patch_maps)  # each (B, 28, 28)
                else:
                    batch_ad_scores = self.predict(imgs) # (B)
                img_ad_scores += batch_ad_scores

                # Handle ground truth masks (can be None for normal samples)
                for i, gt_mask in enumerate(data['ground_truth_mask']):
                    if gt_mask is not None:
                        # Ensure mask has same number of channels as image
                        if gt_mask.shape[0] == 1:  # Single channel mask
                            gt_mask = gt_mask.expand(3, -1, -1)  # Expand to 3 channels
                        gt_masks_list.append(gt_mask)
                        labels.append(1)
                    else:
                        # Create zero mask for normal samples (same shape as corresponding image)
                        zero_mask = torch.zeros_like(imgs[i].cpu())
                        gt_masks_list.append(zero_mask)
                        labels.append(0)

        ad_scores = torch.tensor(img_ad_scores) # N
        gt_masks = torch.cat(gt_masks_list, dim=0) # legacy shape (3N, 224, 224); see docstring.
        labels = torch.tensor(labels) # N

        if return_patch_scores:
            patch_maps = torch.cat(patch_maps_list, dim=0)  # (N, 28, 28)
            return ad_scores, gt_masks, labels, patch_maps, gt_masks_list
        return ad_scores, gt_masks, labels

    def calc_percentiles(self, train_dataloader, percentile=0.95):
        """
        For testing, evaluates thresholds for reconstruction error of normal samples
        """
        img_ad_scores, gt_masks, labels = self.eval_one_epoch(train_dataloader)

        # Returns one float, the threshold at which 95% (percentile) amount of normal
        # samples are below the final AD score
        return torch.quantile(img_ad_scores, percentile).item()

    def calc_results(self, dataloader,
                     dataset, task, all_tasks, exp,
                     metrics, final, **kwargs):
        """
        Calculate results of the model on a testing set.
        Args:
            dataloader: Dataloader for the testing data.
            dataset: (str) name of the dataset, either 'MVTEC' or 'MTD'
            task: (str) name of the task, used for column indexing
            all_tasks: (list[str]) list of all task names for column naming
            exp: (str) name of the experiment, either 'unsupervised' or 'supervised'
            metrics: (list) list of metrics to calculate for each task
            final: (bool) whether this calculation is done using the final model or not
            kwargs:train_dataloader: (torch.utils.data.DataLoader) training data loader used to calculate
                                reconstruction loss for normal samples. Assumes this training data
                                is only normal data, as reconstruction is inherently unsupervised
        Returns:
            Nothing; saves df's to appropriate files
        """
        # Shapes (N, C, H, W) and (N)
        # recon_error is in [0, 1] and labels predicts whether image has anomaly (1) or not (0)
        # If pixel_auroc is requested, ask eval_one_epoch for per-patch maps too.
        need_pixel = "pixel_auroc" in metrics
        if need_pixel:
            img_ad_scores, gt_masks, labels, patch_maps, gt_masks_raw = \
                self.eval_one_epoch(dataloader, return_patch_scores=True)
        else:
            img_ad_scores, gt_masks, labels = self.eval_one_epoch(dataloader)
        threshold = self.calc_percentiles(kwargs.get("train_dataloader"))
        preds = (img_ad_scores > threshold).int()

        # Pre-compute pixel-level scores at 224x224 (matches GT mask resolution).
        pixel_scores_224 = None
        gt_masks_binary_224 = None
        if need_pixel:
            # patch_maps: (N, 28, 28) -> upsample to (N, 224, 224)
            pixel_scores_224 = upsample_anomaly_map(patch_maps, target_size=224)
            gt_masks_binary_224 = stack_gt_masks(gt_masks_raw, target_size=224)

        # Go through metrics for current dataset and experiment
        result_files = os.listdir("results/eval_metrics") # assumes this is run from the root folder src
        for m in metrics:
            filename = f"MTD_{task.split("_")[0]}_{exp}_{m}.csv" if dataset == "MTD" else f"{dataset}_{exp}_{m}.csv"
            # Check if csv file exists.
            if filename in result_files:
                # If it does exist, load it in
                df = pd.read_csv(os.path.join("results/eval_metrics", filename), index_col=0)
            else:
                # if it doesn't, use function to create it
                df = pd.DataFrame(columns=all_tasks)
                df.index.name = "Model"  # Give the index a name for clarity

            # Perform calculations for that metric here
            # 1 = anomaly, 0 = good
            model_name = "Patchcore_final" if final else "Patchcore"
            if m == "img_acc":
                acc = ((preds==labels).sum() / len(preds)).item()
                df.loc[model_name, task] = acc
            elif m == "img_recall":
                recall = (((preds==1)*(labels==1)).sum() / ((labels==1).sum())).item()
                df.loc[model_name, task] = recall
            elif m == "img_auroc":
                df.loc[model_name, task] = tef.binary_auroc(img_ad_scores, labels).item()
            elif m == "pixel_auroc":
                df.loc[model_name, task] = compute_pixel_auroc(
                    pixel_scores_224, gt_masks_binary_224
                )

            # Update DF and save it
            df.to_csv(os.path.join("results/eval_metrics", filename))

        return

    def save(self, model_path: str):
        """
        Save PatchCore model including backbone, projection weights, and memory bank

        :param path: filepath to save to (without extension)
        """

        # Prepare save dictionary
        save_dict = {
            # Model architecture info
            'backbone_name': self.backbone.backbone_name,
            'coreset_pct': self.coreset_pct,

            # Model weights
            'backbone_state_dict': self.backbone.state_dict(),
            'projection_weights': self.random_projection.state_dict(),

            # Memory bank (the core of PatchCore)
            'memory_bank': self.memory_bank.cpu(),

            # Device info for proper loading
            'device': str(self.backbone.device)
        }

        # Save as .pth file
        torch.save(save_dict, model_path)

        return

    def load(self, path: str):
        """
        Load PatchCore model

        :param path: filepath to load from (without extension)
        """
        # Load the saved dictionary
        save_dict = torch.load(path, map_location=self.device)

        # Restore memory bank
        if save_dict['memory_bank'] is not None:
            self.memory_bank = save_dict['memory_bank'].to(self.backbone.device)
        else:
            self.memory_bank = None
            print("Warning: No memory bank found. Model needs training.")

        # Restore model weights
        self.backbone.load_state_dict(save_dict['backbone_state_dict'])
        self.random_projection.load_state_dict(save_dict['projection_weights'])

        self.to(self.device)
        return
