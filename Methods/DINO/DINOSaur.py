import torch
import torch.nn as nn
import torch.nn.functional as F
import torcheval.metrics.functional as tef
from einops import rearrange
import numpy as np
import os
import pandas as pd
from Methods import (
    BaseAnomalyDetector,
    compute_pixel_auroc,
    upsample_anomaly_map,
    stack_gt_masks,
)
from .get_dino_model import get_dino_model

class DINOSaur_Model(BaseAnomalyDetector):
    """
    DINOSaur: DINO Spatial Anomaly Unsupervised Recognition

    Algorithm Notes:
        Available DINOv3 variants:
            - dinov3_vits16
            - dinov3_vits16plus
            - dinov3_vitb16
            - dinov3_vitl16

            - dinov3_convnext_tiny
            - dinov3_convnext_small
            - dinov3_convnext_base
            - dinov3_convnext_large
    """
    def __init__(self,
                 dino_variant: str = 'dinov3_vits16',
                 weights_dir="Methods/DINO/dinov3_weights",
                 coreset_pct=0.2,
                 neighborhood=3
                 ):
        assert 0.0 < coreset_pct <= 1.0, "coreset_pct should be between 0 and 1, inclusive"
        ### Our modules
        self.dino_variant = dino_variant
        self.weights_dir = weights_dir
        self.neighborhood = neighborhood

        # Get device and move all modules there
        super().__init__()

        # Create memory banks, both for class token means, and patches
        self.cls_memory = {} # dict of tensors: (T, 384), where T is number of tasks
        self.patch_memory = {} # dict of tensors: (14, 14, M, 384), where M is coreset pct of all patch tokens
        self.coreset_pct = coreset_pct

        # Get dino variant
        self.dino = get_dino_model(dino_variant, weights_dir)
        for param in self.dino.parameters():
            param.requires_grad = False
        self.dino.to(self.device)

        return

    def forward(self, x: torch.Tensor):
        """
        Passes batched images and gets final features for inference or training.
        :param x: image of size (B, C, 224, 224)
        :return: two tensors,
            - cls token (B, 384)
            - patch tokens (B, 196, 384)
        """
        with torch.no_grad():
            out = self.dino.forward_features(x)
            cls_token = out['x_norm_clstoken'] # (B, 384)
            patch_tokens = out['x_norm_patchtokens'] # (B, 196, 384)

        return cls_token, patch_tokens

    def _coreset_sampling(self, task_features: torch.Tensor):
        """
        Applies coreset sampling to all patch features for the given task features, which
        are a tensor of size (D*196, 384), and D is the dataset size

        It will return  a tensor of size (M, 384), where M = D*196*coreset_pct
        """
        # Memory bank size and corresponding coreset size
        m_size = task_features.shape[0]
        coreset_size = max(1000, int(m_size * self.coreset_pct))
        # Chosen coreset indices
        coreset_idx = torch.zeros(m_size).to(self.device)
        # randomly choose first index and calculate max distance
        random_idx = torch.randint(low=0, high=m_size, size=(1,1)).item()
        coreset_idx[random_idx] = 1
        chosen_feature = task_features[random_idx]
        min_dist = F.pairwise_distance(task_features, chosen_feature).to(self.device)
        while coreset_idx.sum() < coreset_size:
            # Get all non_chosen indices and zero out any already chosen indices' distance
            # So, it's not calculated for finding next max distance
            non_chosen_idx = abs(coreset_idx - 1).to(self.device)
            min_dist *= non_chosen_idx
            # Get new coreset point
            new_idx = min_dist.argmax()
            coreset_idx[new_idx] = 1
            chosen_feature = task_features[new_idx]
            # Calculate new minimum distance to new feature
            new_min_dist = F.pairwise_distance(task_features, chosen_feature).to(self.device)
            # If the new distance is less than the old distance, update min_dist
            min_dist = torch.min(min_dist, new_min_dist)

        # Get final coreset
        task_features = task_features[coreset_idx.bool().cpu()]

        return task_features

    def _patch_coreset_sampling(self, task_features: torch.Tensor):
        """
        Applies coreset sampling to each patch location for the given task features, which
        are a tensor of size (D, 196, 384), and D is the dataset size

        It will return  a tensor of size (14, 14, M, 384), where M = D*coreset_pct
        """
        # Memory bank size and corresponding coreset size
        d_size = task_features.shape[0]
        coreset_size = max(20, int(d_size * self.coreset_pct))
        num_patches = task_features.shape[1] # 196
        coreset_patch_features = []

        # Do coreset sampling for each patch
        for p in range(num_patches):
            chosen_idx = torch.zeros(d_size).to(self.device) # (D)
            # Get this patch's features
            patch_features = task_features[:, p, :].to(self.device) # (D, 384)
            # Start with first randomly chosen point
            random_idx = torch.randint(low=0, high=d_size, size=(1,1)).item()
            chosen_idx[random_idx] = 1
            coreset_dists = F.pairwise_distance(patch_features, patch_features[random_idx]).to(self.device) # D
            while chosen_idx.sum() < coreset_size:
                # Get all non_chosen indices and zero out any already chosen indices' distance
                # So, it's not calculated for finding next max distance
                non_chosen_idx = abs(chosen_idx - 1).to(self.device)
                coreset_dists *= non_chosen_idx
                # Get new coreset point
                new_idx = coreset_dists.argmax()
                chosen_idx[new_idx] = 1
                chosen_feature = patch_features[new_idx]
                # Calculate new minimum distance to new feature
                new_min_dist = F.pairwise_distance(patch_features, chosen_feature).to(self.device)
                # If the new distance is less than the old distance, update min_dist
                coreset_dists = torch.min(coreset_dists, new_min_dist)

            chosen_features = patch_features[chosen_idx.bool(), :] # (M, 384)
            coreset_patch_features.append(chosen_features)


        coreset_patch_features = torch.stack(coreset_patch_features) # (num_patches, coreset_size, 384)
        coreset_patch_features = rearrange(coreset_patch_features,
                                           '(w h) c e -> w h c e',
                                           w=int(np.sqrt(num_patches)))
        return coreset_patch_features # (14, 14, M, 384)

    def train_one_epoch(self, dataloader: torch.utils.data.dataloader,
                        task_name: str,
                        **kwargs):
        """
        Given a dataloader of nominal data, fits the model to memory bank,
        then applies coreset sampling

        :param dataloader: dataloader of nominal data
        """
        with torch.no_grad():

            task_cls_tokens = []
            task_patch_tokens = []

            for batch_idx, data in enumerate(dataloader):

                # Get initial features
                imgs = data['image'].to(self.device) # (B, 3, 224, 224)
                cls_token, patch_tokens = self.forward(imgs)

                # Add to memory
                task_cls_tokens.append(cls_token.detach().cpu())
                task_patch_tokens.append(patch_tokens.detach().cpu())

            task_cls_tokens = torch.cat(task_cls_tokens, dim=0) # (D, 384), D = dataset len
            task_patch_tokens = torch.cat(task_patch_tokens, dim=0)  # (D, 196, 384)

            # Consolidate memory banks
            task_cls_token_prototype = task_cls_tokens.mean(dim=0) # (384)
            self.cls_memory[task_name] = task_cls_token_prototype

            # task_memory_features = self._coreset_sampling(task_patch_tokens)
            task_memory_features = self._patch_coreset_sampling(task_patch_tokens) # (14, 14, M, 384)
            # Turns memory bank to size (M_coreset, embed_dim)
            self.patch_memory[task_name] = task_memory_features

        return 0

    def predict(self, x: torch.Tensor, neighborhood: int):
        """
        Vectorized prediction of patch-level anomaly scores.

        :param x: batched image tensor of size (B, 3, 224, 224)
        :param neighborhood: neighborhood size (radius) for which to compare patches, in [0, 13]
        :return: a tensor of patch-level ad scores (B, 14, 14)
        """
        with torch.no_grad():
            P = 14  # num_patches per side
            embed_dim = 384
            patch_core_ad_scores = torch.zeros(x.shape[0], P, P).to(self.device)
            batch_cls_tokens, batch_patch_tokens = self.forward(x.to(self.device))

            for b in range(x.shape[0]):
                cls_token = batch_cls_tokens[b]  # (384,)
                patch_tokens = batch_patch_tokens[b]  # (196, 384)
                patch_tokens = rearrange(patch_tokens, '(w h) c -> w h c', w=P)  # (14, 14, 384)

                # Task Identification — vectorized over all tasks
                task_keys = list(self.cls_memory.keys())
                prototypes = torch.stack([self.cls_memory[t].to(self.device) for t in task_keys])  # (T, 384)
                dists = torch.norm(prototypes - cls_token.unsqueeze(0), dim=1)  # (T,)
                chosen_task = task_keys[dists.argmin().item()]

                # Image prediction — vectorized using padding + unfold
                # task_features: (14, 14, M, 384) where M = coreset size for this task
                task_features = self.patch_memory[chosen_task].to(self.device)
                M = task_features.shape[2]

                # Rearrange to (M*384, 14, 14) so we can use F.pad and unfold on spatial dims
                tf_flat = rearrange(task_features, 'h w m c -> (m c) h w')  # (M*384, 14, 14)

                # Pad spatial dims with inf so padded entries never win the min distance
                tf_padded = F.pad(tf_flat, (neighborhood, neighborhood, neighborhood, neighborhood),
                                  value=float('inf'))
                # tf_padded: (M*384, 14+2n, 14+2n)

                # Unfold to extract all (2n+1)×(2n+1) windows at each position
                ws = 2 * neighborhood + 1  # window size
                # unfold(dim, size, step) — extract sliding windows along spatial dims
                windows = tf_padded.unfold(1, ws, 1).unfold(2, ws, 1)
                # windows: (M*384, 14, 14, ws, ws)

                # Rearrange back to (14, 14, ws*ws*M, 384)
                windows = rearrange(windows, '(m c) h w wh ww -> h w (wh ww m) c',
                                    m=M, c=embed_dim)

                # Compute distances: for each patch position, distance to all neighborhood memory patches
                # patch_tokens: (14, 14, 384) → (14, 14, 1, 384)
                # windows:      (14, 14, ws*ws*M, 384)
                diffs = patch_tokens.unsqueeze(2) - windows  # (14, 14, ws*ws*M, 384)
                dists = torch.norm(diffs, dim=3)  # (14, 14, ws*ws*M)

                # Inf-padded entries produce inf distance, so min() ignores them
                patch_core_ad_scores[b] = dists.min(dim=2).values  # (14, 14)

        return patch_core_ad_scores  # (B, 14, 14)

    def eval_one_epoch(self, dataloader, neighborhood):
        """
        Evaluate model on one epoch, collecting raw model outputs for later analysis.

        Returns:
            patch_ad_scores: (N, 14, 14) float — per-patch anomaly scores at the
                             native 14x14 grid resolution.
            img_ad_scores:   (N,) float — image-level scores (max over patches).
            gt_masks:        (N, 224, 224) tensor (still ImageNet-normalized) —
                             kept for backward compat with any external caller.
            labels:          (N,) int binary labels.
            gt_masks_raw:    list of N tensors — raw per-image GT masks as
                             returned by the dataset (still normalized). Use
                             stack_gt_masks() in calc_results to recover binary.
                             None entries for normal samples are replaced with
                             explicit zero tensors (1, 224, 224) so downstream
                             helpers can resize them.
        """
        self.eval()

        patch_ad_scores = []
        gt_masks_list = []
        labels = []

        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch_idx, data in enumerate(dataloader):

                # Get ad scores from the predict() method
                imgs = data['image'].to(self.device) # (B, 3, 224, 224)

                batch_patch_core_ad_scores = self.predict(imgs, neighborhood) # (B, 14, 14)
                patch_ad_scores.append(batch_patch_core_ad_scores)

                # Handle ground truth masks (can be None for normal samples)
                for i, gt_mask in enumerate(data['ground_truth_mask']):
                    if gt_mask is not None:
                        gt_masks_list.append(gt_mask)
                        labels.append(1)
                    else:
                        # Create zero mask for normal samples (same shape as corresponding image)
                        zero_mask = torch.zeros(1, 224, 224).cpu()
                        gt_masks_list.append(zero_mask)
                        labels.append(0)

        patch_ad_scores = torch.cat(patch_ad_scores, dim=0) # (N, 14, 14)
        img_ad_scores = torch.tensor([fm.max().item() for fm in patch_ad_scores]).to(self.device)
        gt_masks = torch.cat(gt_masks_list, dim=0) # (N, 224, 224) — each item was (1, 224, 224)
        labels = torch.tensor(labels).to(self.device) # N

        return patch_ad_scores, img_ad_scores, gt_masks, labels, gt_masks_list

    def calc_percentile(self, train_dataloader, neighborhood):
        """
        For testing, evaluates thresholds for ad scores (distances) of normal samples and calculates img_ad_scores
        """
        # Note: eval_one_epoch now returns a 5-tuple (added gt_masks_list for pixel metrics).
        # We only need img_ad_scores here.
        _, img_ad_scores, _, _, _ = self.eval_one_epoch(train_dataloader, neighborhood)
        threshold = torch.quantile(img_ad_scores, q=0.975)

        return threshold

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
        # labels predicts whether image has anomaly (1) or not (0)
        patch_ad_scores, img_ad_scores, gt_masks, labels, gt_masks_raw = \
            self.eval_one_epoch(dataloader, self.neighborhood)
        train_dataloader = kwargs.get("train_dataloader")
        img_score_threshold = self.calc_percentile(train_dataloader, self.neighborhood)
        preds = (img_ad_scores > img_score_threshold).int()

        # Pre-compute pixel-level scores once (used only if 'pixel_auroc' is in metrics)
        # patch_ad_scores: (N, 14, 14) -> upsample to (N, 224, 224) to match GT mask resolution
        pixel_scores_224 = None
        gt_masks_binary_224 = None
        if "pixel_auroc" in metrics:
            pixel_scores_224 = upsample_anomaly_map(patch_ad_scores.cpu(), target_size=224)
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

            model_name = f"DINOSaur"
            if final:
                model_name = model_name + "_final"

            # Perform calculations for that metric here
            # 1 = anomaly, 0 = good
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

        :param path: filepath to save to
        """

        # Prepare save dictionary
        save_dict = {
            # Model architecture info
            'dino variant': self.dino_variant,
            'weights directory': self.weights_dir,
            'coreset pct': self.coreset_pct,

            # Memory banks
            'class token prototype memory': self.cls_memory,
            'patch token memory bank': self.patch_memory,

        }

        # Save as .pth file
        torch.save(save_dict, model_path)

        return

    def load(self, path: str):
        """
        Load PatchCore model

        :param path: filepath to load from
        """
        # Load the saved dictionary
        save_dict = torch.load(path, map_location=self.device)

        # Load model
        self.dino = get_dino_model(save_dict['dino variant'],
                                   save_dict['weights directory'])
        for param in self.dino.parameters():
            param.requires_grad = False

        # Restore memory
        self.cls_memory = save_dict['class token prototype memory']
        self.patch_memory = save_dict['patch token memory bank']

        # Set up device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"Using {self.device} device")

        self.dino.to(self.device)

        return
