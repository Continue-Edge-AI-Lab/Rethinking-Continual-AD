import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from einops import rearrange
import numpy as np
import os
import pandas as pd
from torch.utils.data import dataloader
import torcheval.metrics.functional as tef
from ultralytics import FastSAM
from Methods import (
    BaseAnomalyDetector,
    compute_pixel_auroc,
    stack_gt_masks,
)
from Methods.UCAD.ViT import ViT
from datasets import transforms


class UCAD_Model(BaseAnomalyDetector):
    """
    UCAD Module

    Uses a single, frozen ViT and a Segment-Anything-Model (SAM) for training.
    For installation, use:
    - pip install ultralytics

    Algorithm Notes:
        - create_key() and create_knowledge() need to be done before and after training, respectively
        - TRAINING:
            - We are just getting loss based on ViT features and segmentation masks.
            - The authors don't mention how they aggregate the downsampled masks, so
                we just take the maximum class
        - During testing, we will have 3 separate steps
            - Before evaluating, calculate key and knowledge with given dataloader
            - Then, during evaluation, calculate results
        - Farthest Point Sampling
            - Takes 196 representative patches, regardless of spatial position, across all images
            - Has a set of "chosen" and "unchosen" points. Iteratively adds points to the "chosen" list
                that are farther from all current points in "chosen"
        - Coreset Sampling
    """
    def __init__(self,
                 vit_output_layer=5, # authors use 5, but may be useful to look at changing
                 inference_only=False # skip FastSAM loading for inference-only mode
                 ):
        super().__init__()
        self.vit_output_layer = vit_output_layer

        # Get base ViT and freeze parameters
        # Notice this is a custom version of PyTorch's ViT for our use
        self.vit = ViT(output_layer=vit_output_layer,
                       device=self.device)
        for param in self.vit.parameters():
            param.requires_grad = False

        # Memory banks,
        # each a list of tensors (size = num_tasks) that correspond to a given task
        self.key_memory = []
        self.prompt_memory = []
        self.knowledge_memory = []

        # Create SAM Model (only needed for training — segmentation is not used during inference)
        if not inference_only:
            self.sam = FastSAM('FastSAM-s.pt')
            for param in self.sam.parameters():
                param.requires_grad = False

        # Transformation to upscale patch-level predictions
        self.upscale_transform =T.Compose([
            T.Resize((224, 224)),
            T.GaussianBlur(kernel_size=5, sigma=0.25)
        ])

        self.register_parameter('prompt',
            nn.Parameter(torch.randn(self.vit_output_layer, 768, device=self.device))
        )

        return

    def forward(self, img_batch, with_prompt):
        # (B, 3, 224, 224)
        img_batch = img_batch.to(self.device)

        # Get ViT output, with or without prompt vector
        # (B, 196, 768)
        out = self.vit(img_batch, self.prompt if with_prompt else None)

        return out

    def segment(self, img_batch):
        # (B, 3, 224, 224)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]

        # Denormalize from ImageNet normalization to [0, 1] range for FastSAM
        # Images come in as: (img - mean) / std, so reverse: img = img * std + mean
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_denorm = img_batch * imagenet_std + imagenet_mean
        img_denorm = torch.clamp(img_denorm, 0.0, 1.0)  # Ensure [0, 1] range

        # Process segments in small chunks to bound peak GPU memory during FastSAM
        # inference. Masks are identical to processing the full batch at once; only
        # the peak allocation changes.
        masks = []
        sam_chunk = 2
        with torch.no_grad():
            for start in range(0, batch_size, sam_chunk):
                out = self.sam.predict(img_denorm[start:start + sam_chunk],
                                       imgsz=224,
                                       retina_masks=True,
                                       conf=0.3,
                                       iou=0.9,
                                       verbose=False,
                                       device=self.device)
                for o in out:
                    # Pool each mask to 14x14 immediately. UCAD only ever uses masks
                    # at 14x14 (see the loss), so the large (M, 224, 224) masks never
                    # leave this function -- a 256x cut in held mask memory.
                    masks.append(F.max_pool2d(o.masks.data, kernel_size=16, stride=16))
                del out  # drop our reference so the Results can be collected
        # Outputs list with length=B of tensors, each of shape [M, 224, 224]
        # where M is the number of masks found
        return masks

    def update_key_memory(self, dataloader):
        """
        Evaluates an entire task's data and uses Farthest point sampling to get representative
        features to be added into key_memory
        """

        key_bank = []
        # Get all key values
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                imgs = data['image'].to(self.device)
                # (B, 197, 768)
                out = self.forward(imgs, with_prompt=False)
                key_bank.append(out.cpu())
        # (num_images*196, 768)
        key_bank = rearrange(torch.cat(key_bank, dim=0), 'n_i n_p d -> (n_i n_p) d')

        # Use Farthest Point Sampling to get representative sample.
        # .clone() is essential: key_bank[idx] returns a VIEW sharing key_bank's
        # storage, and key_bank is rebuilt every iteration, so keeping the view
        # would pin every generation of the bank (gigabytes each) in memory.
        chosen_patches = [key_bank[0].clone()]

        key_bank = key_bank[1:, :]
        num_patches = key_bank.shape[0]
        min_dists = (torch.ones(num_patches).float()*torch.inf).to(self.device)
        # Chose first patch to start, then add 195 more iteratively
        for i in range(1, 196):
            # Only need to update distance from most recent addition
            new_min_dists = F.pairwise_distance(key_bank, chosen_patches[-1]).to(self.device)
            min_dists = torch.min(new_min_dists, min_dists)

            # Extract patch with maximum distance
            chosen_idx = min_dists.argmax()
            chosen_patches.append(key_bank[chosen_idx].clone())

            # Remove chosen sample from remaining key bank
            remaining_idx = torch.ones(key_bank.shape[0], dtype=torch.bool)
            remaining_idx[chosen_idx] = False
            key_bank = key_bank[remaining_idx]
            min_dists = min_dists[remaining_idx]

        # Add these patches to long-term key memory
        self.key_memory.append(torch.stack(chosen_patches))

        return

    def update_knowledge_memory(self, dataloader):
        """
        Evaluates an entire task's data and uses Coreset sampling to get representative
        features to be added into knowledge_memory.

        Paper Equation 2: K_n = CoreSetSampling(k^i)
        Coreset sampling minimizes max distance from any point to its nearest selected point,
        which better represents the "normal" feature distribution than FPS.
        """

        knowledge_bank = []
        # Get all knowledge values
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                imgs = data['image'].to(self.device)
                # (B, 196, 768)
                out = self.forward(imgs, with_prompt=True)
                knowledge_bank.append(out.cpu())
        # (num_images*196, 768)
        knowledge_bank = rearrange(torch.cat(knowledge_bank, dim=0), 'n_i n_p d -> (n_i n_p) d')

        # Use Coreset Sampling (Greedy K-Center) per Paper Equation 2
        # This minimizes the maximum distance from any point to its nearest selected point
        num_to_select = 196
        num_total = knowledge_bank.shape[0]

        # Initialize with a random point (or first point)
        selected_indices = [0]
        # Track minimum distance from each point to any selected point
        min_dists_to_selected = torch.full((num_total,), float('inf'))

        for _ in range(1, num_to_select):
            # Update distances based on most recently added point
            last_selected = knowledge_bank[selected_indices[-1]]
            dists_to_last = torch.norm(knowledge_bank - last_selected.unsqueeze(0), dim=1)
            min_dists_to_selected = torch.min(min_dists_to_selected, dists_to_last)

            # Mask already selected points
            min_dists_to_selected[selected_indices] = -1

            # Select point with maximum distance to its nearest selected point (greedy k-center)
            next_idx = min_dists_to_selected.argmax().item()
            selected_indices.append(next_idx)

        # Extract selected patches
        chosen_patches = knowledge_bank[selected_indices]

        # Add these patches to long-term knowledge memory
        self.knowledge_memory.append(chosen_patches)

        return

    def update_memory(self, dataloader):
        """
        Updates the key, prompt, and knowledge memory after each task
        """
        self.update_key_memory(dataloader)
        self.prompt_memory.append(self.get_parameter('prompt').clone().detach())
        self.update_knowledge_memory(dataloader)
        # Offload the stored per-task banks to CPU and clear the GPU cache so memory
        # does not accumulate on the GPU across tasks during training. Eval reloads
        # the banks to the GPU from disk, so results are unaffected.
        self.key_memory = [t.cpu() for t in self.key_memory]
        self.prompt_memory = [t.cpu() for t in self.prompt_memory]
        self.knowledge_memory = [t.cpu() for t in self.knowledge_memory]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    def train_one_epoch(self, dataloader,
                        optimizer,
                        **kwargs):
        """
        Train a model on one epoch.
        Args:
            dataloader: Dataloader for the training data.
            optimizer: torch.optim.Optimizer
            kwargs: Additional keyword arguments

        Returns: epoch_loss, the total accumulated loss for that epoch
        """
        self.prompt.requires_grad_(True)

        epoch_loss = 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            imgs = data['image'].to(self.device)

            # (B, 196, 768)
            out = self.forward(imgs, with_prompt=True)
            # list of len B, each being (M, 224, 224), M binary masks for different objects
            masks = self.segment(imgs)
            loss = UCAD_Contrastive_loss(out, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Drop per-batch references and periodically collect, so ultralytics
            # Results reference cycles do not accumulate in RAM across batches.
            del out, masks, loss
            if (batch_idx + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return epoch_loss

    def eval_one_epoch(self, dataloader, **kwargs):
        """
        Evaluate model on one epoch, collecting raw model outputs for later analysis.

        This method implements the UCAD test-time inference algorithm following the paper:
        1. Task Selection: Find closest key to identify the appropriate task
        2. Feature Extraction: Use corresponding prompt to get task-adapted features
        3. Anomaly Scoring: Compare test features with stored knowledge using patch-level distances
        4. Aggregation: Calculate image-level scores and pixel-level segmentation maps

        Following UCAD paper equations (4), (5), and (6) for task selection and anomaly scoring.

        Args:
            dataloader: Dataloader for the evaluation data
            **kwargs: Additional keyword arguments

        Returns:
            img_scores: Tensor of image-level anomaly scores, shape (D,)
            pixel_scores: Tensor of pixel-level anomaly maps, shape (D, H, W) = (D, 224, 224)
            gt_masks: Tensor of ground truth masks, shape (D, H, W) = (D, 224, 224)
            labels: List of ground truth labels (0=normal, 1=anomaly), (D,)

        Where:
            D = total number of samples in the dataset
            C = number of channels (3 for RGB)
            H, W = image height and width (224, 224)
        """

        # Initialize lists to collect all outputs
        img_scores = []         # Image-level anomaly scores
        pixel_scores = []       # Pixel-level anomaly scores (224x224 full resolution)
        gt_masks = []           # Stacked single-channel masks (legacy return)
        gt_masks_raw_list = []  # Raw per-image masks (full shape, with None replaced by zeros)
                                 # — used by stack_gt_masks() in calc_results to recover binary masks.
        labels = []             # Ground truth labels

        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch_idx, data in enumerate(dataloader):
                # Add data to gt_masks and labels
                for mask in data['ground_truth_mask']:
                    if mask is None:
                        gt_masks.append(torch.zeros(224, 224))
                        gt_masks_raw_list.append(torch.zeros(1, 224, 224))
                    else:
                        gt_masks.append(mask[0,:,:])
                        gt_masks_raw_list.append(mask)
                for l in data['label']:
                    labels.append(l)


                imgs = data['image'].to(self.device)  # (B, 3, 224, 224)

                # Extract features without prompts for initial task identification
                # Following paper: "an image x_test initially locates its corresponding task"
                img_features = self.forward(imgs, with_prompt=False)  # (B, 196, 768)

                # Process each image individually for task identification and anomaly scoring
                for b in range(img_features.shape[0]):
                    img = imgs[b].unsqueeze(0)  # (1, 3, 224, 224)
                    embed = img_features[b, :, :]  # (196, 768) - current image patches
                    sim_scores = []

                    # Calculate similarity with each key for task identification
                    for k in range(len(self.key_memory)):
                        test_key = self.key_memory[k]  # (196, 768) - stored key for task k

                        # Calculate similarity following Equation 4
                        # Sum over all image patches: for each test patch, find min distance to key patches
                        sim = 0
                        for patch in range(196):  # N_p = 196 patches per image
                            # Find minimum distance from current patch to all patches in the key
                            patch_dists = F.pairwise_distance(
                                embed[patch].unsqueeze(0),  # (1, 768)
                                test_key.to(self.device)    # (196, 768)
                            )
                            sim += patch_dists.min()  # Add minimum distance

                        sim_scores.append(sim)

                    # Select task with minimum similarity score (arg min)
                    chosen_task = torch.stack(sim_scores).argmin().item()

                    # Extract task-adapted features using the corresponding key
                    self.prompt = nn.Parameter(self.prompt_memory[chosen_task].clone().detach())
                    img_knowledge = self.forward(img, with_prompt=True).squeeze(0)  # (196, 768)
                    task_knowledge = self.knowledge_memory[chosen_task].to(self.device)  # (196, 768)

                    # =====================================================
                    # Anomaly Scoring following Paper Equations 5 and 6
                    # =====================================================

                    # Step 1: For each test patch, find nearest knowledge patch and compute base distance
                    # Paper Eq. 5: m*, m_test,* = argmax argmin ||m_test - m||_2
                    m_star_idx = []   # Index of closest knowledge patch for each test patch
                    m_star_dist = []  # Base distance (s*_i) for each test patch

                    for patch in range(196):
                        test_patch = img_knowledge[patch]  # (768,) - current test patch

                        # Find closest knowledge patch
                        dists_to_knowledge = F.pairwise_distance(
                            task_knowledge,  # (196, 768)
                            test_patch,      # (768,)
                        )
                        min_dist_idx = dists_to_knowledge.argmin().item()
                        min_dist = dists_to_knowledge.min().item()

                        m_star_idx.append(min_dist_idx)
                        m_star_dist.append(min_dist)

                    m_star_idx = torch.tensor(m_star_idx).to(self.device)
                    # s*_i = ||m_test_i - m*_i||_2 (base anomaly score for each patch)
                    base_scores = torch.tensor(m_star_dist).to(self.device)

                    # Step 2: Apply re-weighting (Paper Equation 6)
                    # s_i = (1 - exp(||m_test_i - m*_i||) / sum_{m in N_b(m*_i)} exp(||m_test_i - m||)) * s*_i
                    #
                    # Key insight: The re-weighting uses distances from the TEST PATCH to neighbors
                    # of its nearest knowledge patch, making anomalies in sparse regions score higher.

                    num_neighbors = 5  # k for nearest neighbors in knowledge bank
                    final_s_scores = []

                    for patch in range(196):
                        test_patch = img_knowledge[patch]  # (768,) - this test patch
                        nearest_knowledge_idx = m_star_idx[patch]
                        nearest_knowledge = task_knowledge[nearest_knowledge_idx]  # (768,)

                        # Find k nearest neighbors of m*_i in the knowledge bank
                        nn_dists_in_knowledge = F.pairwise_distance(
                            task_knowledge,      # (196, 768)
                            nearest_knowledge,   # (768,)
                        )
                        _, neighbor_indices = torch.topk(nn_dists_in_knowledge, k=num_neighbors, largest=False)
                        neighbors = task_knowledge[neighbor_indices]  # (k, 768)

                        # Compute distances from THIS test patch to neighbors of its nearest knowledge
                        dists_to_neighbors = F.pairwise_distance(
                            neighbors,           # (k, 768)
                            test_patch,          # (768,)
                        )

                        # Apply Equation 6 re-weighting
                        numerator = torch.exp(base_scores[patch])
                        denominator = torch.exp(dists_to_neighbors).sum()

                        # Avoid division by zero
                        denominator = torch.clamp(denominator, min=1e-5)

                        # Final re-weighted score for this patch
                        weight = 1.0 - (numerator / denominator)
                        final_s = weight * base_scores[patch]
                        final_s_scores.append(final_s)

                    # (196,) - per-patch anomaly scores
                    final_s_scores = torch.stack(final_s_scores).to(self.device)
                    # (14, 14)
                    patch_scores = rearrange(final_s_scores, '(np1 np2) -> np1 np2', np1=14)

                    # Convert anomaly scores into pixel and img-level predictions
                    # (224, 224)
                    pixel_score = self.upscale_transform(patch_scores.unsqueeze(0)).squeeze(0)
                    img_score = patch_scores.max().item()  # S_img = max(s_i), i ∈ N_p

                    img_scores.append(img_score)
                    pixel_scores.append(pixel_score)

        # D = length of dataset
        labels = torch.tensor(labels).cpu()            # (D)
        gt_masks = torch.stack(gt_masks).cpu()         # (D, 224, 224)
        pixel_scores = torch.stack(pixel_scores).cpu() # (D, 224, 224)
        img_scores = torch.tensor(img_scores).cpu()    # (D)

        # Note: gt_masks_raw_list returned for downstream pixel-AUROC computation;
        # call stack_gt_masks() on it to get a binary (D, 224, 224) tensor.
        return img_scores, pixel_scores, gt_masks, labels, gt_masks_raw_list

    def calc_percentiles(self, train_dataloader, percentile=0.975):
        """
        For testing, evaluates thresholds for reconstruction error of normal samples
        """
        # eval_one_epoch now returns a 5-tuple (added gt_masks_raw_list).
        img_scores, pixel_scores, gt_masks, labels, _ = self.eval_one_epoch(train_dataloader)

        img_lvl_threshold = torch.quantile(img_scores, percentile).item()
        # Returns one float, the threshold at which 95% (percentile) amount of normal
        # samples are below in recon_error
        return img_lvl_threshold

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
        img_scores, pixel_scores, gt_masks, labels, gt_masks_raw = self.eval_one_epoch(dataloader)
        img_lvl_threshold = self.calc_percentiles(kwargs.get("train_dataloader"))
        preds = (img_scores > img_lvl_threshold).int()

        # Pre-compute pixel-AUROC inputs at native UCAD resolution (224x224).
        gt_masks_binary_224 = None
        if "pixel_auroc" in metrics:
            # pixel_scores already at 224x224 — no upsampling required.
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
            model_name = "UCAD_final" if final else "UCAD"
            if m == "img_acc":
                acc = ((preds==labels).sum() / len(preds)).item()
                df.loc[model_name, task] = acc
            elif m == "img_recall":
                recall = (((preds==1)*(labels==1)).sum() / ((labels==1).sum())).item()
                df.loc[model_name, task] = recall
            elif m == "img_auroc":
                df.loc[model_name, task] = tef.binary_auroc(img_scores, labels).item()
            elif m == "pixel_auroc":
                df.loc[model_name, task] = compute_pixel_auroc(
                    pixel_scores, gt_masks_binary_224
                )

            # Update DF and save it
            df.to_csv(os.path.join("results/eval_metrics", filename))

        return

    def save(self, model_path):
        """
        Saves the model to disk
        Args:
            model_path: a string of the model path
        """
        # We only need to save our memory bank, as the FastSAM and ViT are pre-trained and frozen
        file_prefix = model_path.split("weights")[0]
        torch.save(self.key_memory, file_prefix + "key_memory.pth")
        torch.save(self.prompt_memory, file_prefix + "prompt_memory.pth")
        torch.save(self.knowledge_memory, file_prefix + "knowledge_memory.pth")

        return

    def load(self, model_path):
        """
        Loads a model from a file
        Args:
            model_path: a string of the model path
        """
        # We only need to load our memory bank, as the FastSAM and ViT are pre-trained and frozen
        file_prefix = model_path.split("weights")[0]
        self.key_memory = torch.load(file_prefix + "key_memory.pth", self.device)
        self.prompt_memory = torch.load(file_prefix + "prompt_memory.pth", self.device)
        self.knowledge_memory = torch.load(file_prefix + "knowledge_memory.pth", self.device)

        return

def UCAD_Contrastive_loss(vit_features,
                          masks,
                          lambda_alpha=1.0,
                          lambda_beta=1.0
                          ):
    """
    Calculates contrastive loss for the given vit_features and the segmentation masks.
    Args:
        - vit_features: Tensor of vit features of shape (B, 196, 768)
        - masks: List of length B, with each item being a
                Tensor of vit masks of shape (M, 224, 224),
                corresponding to M binary masks on the image
        - lambda_alpha: float, parameter of the contrastive loss
        - lambda_beta: float, parameter of the contrastive loss
    """
    batch_size = vit_features.shape[0]

    # Masks already arrive pooled to (M, 14, 14) from segment(), so no further
    # downsampling is needed here.
    downsampled_masks = masks

    # Turns downsampled masks to one mask of shape (B, 14, 14), where each image has one segmentation mask
    aggregated_masks = []
    for b in range(batch_size):
        d_mask = downsampled_masks[b] # (M, 14, 14)
        num_masks = d_mask.shape[0]
        for m in range(1, num_masks):
            d_mask[m] *= (m+1)
        d_mask = torch.max(d_mask, 0).values
        aggregated_masks.append(d_mask)
    # (B, 14, 14)
    aggregated_masks = torch.stack(aggregated_masks)

    # Turn mask into tensor of size (B, 196), so each patch has a segmentation label
    aggregated_masks = rearrange(aggregated_masks, 'b h w -> b (h w)')

    # Calculate cosine similarities and add to respective neg/pos losses
    # Negative and positive samples loss
    neg = 0
    num_neg = 0
    pos = 0
    num_pos = 0
    for b in range(batch_size):
        # Calculate cosine similarity for each batch
        labels = aggregated_masks[b] # 196
        features = vit_features[b] # (196, 768)
        norm_features = F.normalize(features, dim=1) # (196, 768)
        cos_sim = torch.matmul(norm_features, norm_features.t()).cpu() # (196, 196)
        for i in range(195):
            f1_label = labels[i]
            f2_labels = labels[i+1:196] # (195-i)

            # Only iterate through upper triangle
            pos_idx = (f2_labels == f1_label).cpu()
            num_pos += pos_idx.sum().item()
            pos += cos_sim[i, i+1:196][pos_idx].sum()
            neg_idx = (f2_labels != f1_label).cpu()
            num_neg += neg_idx.sum().item()
            neg += cos_sim[i, i+1:196][neg_idx].sum()

    # Normalize by number of pairs to get average similarity
    # Paper Equation 3: L_total = λ_α * L_neg_con - λ_β * L_pos_con
    # We normalize to get mean similarity rather than sum
    if num_pos > 0:
        pos /= num_pos
    if num_neg > 0:
        neg /= num_neg

    return (lambda_alpha * neg) - (lambda_beta * pos)

