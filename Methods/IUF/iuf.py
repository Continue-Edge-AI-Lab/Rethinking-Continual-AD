# Methods/IUF/iuf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from torch.utils.data import dataloader
from torchvision.models.vision_transformer import vit_b_16
import torcheval.metrics.functional as tef
from Methods import (
    BaseAnomalyDetector,
    compute_pixel_auroc,
    stack_gt_masks,
)
from Methods.IUF.utils.discriminator import Discriminator
from Methods.IUF.utils.encoder import Encoder
from Methods.IUF.utils.decoder import Decoder

class IUF_Model(BaseAnomalyDetector):
    """
    Complete IUF Module

    We have one ViT class that we build on,
    From there, we will add methods to create the discriminator, encoder, and decoder.

    Algorithm Notes:

        ViT
        - Added positional encoding to the tokens
        - Changed BatchNorm to LayerNorm and ReLU to GELU, in accordance with the ViT paper
        - I noticed that in their Multi-head attention, the authors only work on row-wise attention
          to simplify their computations, so I am going to operate on patches, as the multiplication
          is too large.
        - Batch size needs to maybe be larger than the embedding dimension?

    Updating Strategy (Paper Sec 4.3):
        The gradient update has two components that only activate AFTER the first task:

        1. V-projection (Eq 10-12): Projects gradients into the channel eigenspace of the
           PREVIOUS task (V^T_old), scales by Omega to suppress updates in channels that
           were important for old objects, then projects back. This prevents overwriting
           the old task's learned semantic patterns.

        2. Weight regularization (Eq 9/13): Pulls current weights toward theta_old, the
           parameter snapshot from the end of the previous task. Conceptually similar to
           EWC (Elastic Weight Consolidation) — prevents weights from drifting too far
           from values that worked for old tasks.

    Usage:
        After training each task, the caller MUST call model.save_task_state() before
        starting the next task. This captures V^T and theta for use as V^T_old and
        theta_old during the next task's training.

        Example:
            for task_num, task_data in enumerate(tasks):
                for epoch in range(num_epochs):
                    model.train_one_epoch(task_data.dataloader, optimizer, task_num + 1)
                model.save_task_state()  # <-- CRITICAL: call after each task
    """
    def __init__(self,
                 in_channels=3,
                 patch_size=16, # 224 should be divisible by patch_size
                 embed_dim=64,
                 num_heads=4, # Should be able to take the sqrt easily
                 num_layers=4,
                 num_tasks=15):
        super().__init__()

        self.embed_dim = embed_dim

        # Discriminator is responsible for predicting the task of a given image, and
        # its intermediary features are combined in the encoder with the query to get
        # "Object-Aware Self Attention" (OASA) modules in the encoder
        self.discriminator = Discriminator(device=self.device,
                                           in_channels=in_channels,
                                           patch_size=patch_size,
                                           embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           num_layers=num_layers,
                                           output_size=num_tasks)
        # Encoder creates our latent space and performs SVD on it,
        # which is used in the gradient update
        self.encoder = Encoder(device=self.device,
                               in_channels=in_channels,
                               patch_size=patch_size,
                               embed_dim=embed_dim,
                               num_heads=num_heads,
                               num_layers=num_layers)
        # The decoder decodes the latent space into a reconstructed image
        self.decoder = Decoder(device=self.device,
                               in_channels=in_channels,
                               patch_size=patch_size,
                               embed_dim=embed_dim,
                               num_heads=num_heads,
                               num_layers=num_layers)

        self.v_old = None
        self.theta_old = {}
        self.omega = torch.arange(1, embed_dim + 1, dtype=torch.float32).log().to(self.device)
        self._encoder_decoder_params = set()
        for name, _ in self.named_parameters():
            if name.startswith('encoder.') or name.startswith('decoder.'):
                self._encoder_decoder_params.add(name)

        # Temporary storage for V from the most recent forward pass.
        # save_task_state() reads this to populate v_old for the next task.
        self._current_v = None

        return

    def save_task_state(self):
        """
        Captures the current task's state for use during the NEXT task's training.
        MUST be called after finishing training on each task, before starting the next.

        Saves two things:
            1. V^T from SVD of the encoder's latent space -> becomes v_old for gradient
               projection during the next task (Paper Eq 10-12)
            2. All encoder/decoder weights -> becomes theta_old for weight regularization
               during the next task (Paper Eq 9/13)
        """
        # Save V^T for gradient projection during next task
        if self._current_v is not None:
            self.v_old = self._current_v.clone().detach()

        # Save encoder/decoder weights for regularization during next task
        self.theta_old = {}
        for name, param in self.named_parameters():
            if name in self._encoder_decoder_params:
                self.theta_old[name] = param.data.clone().detach()

        return

    def update_grad(self, loss, beta=0.5):
        """
        Regularized gradient update per Paper Section 4.3.

        Two components, both only active after the first task (when v_old and
        theta_old have been populated by save_task_state()):

        Component 1 — V-projection (Eq 10-12):
            Projects each parameter's gradient into the channel eigenspace of the
            previous task using V^T_old, scales by Omega(k,c) = k*log(c) to suppress
            updates in channels important to old objects, then projects back.

            Math for parameters with last dim = E (embed_dim):
                grad_eigen = grad @ V           (project into eigenspace)
                grad_scaled = omega * grad_eigen (suppress important channels)
                grad_star = grad_scaled @ V^T    (project back)

            Where V = v_old^T and V^T = v_old, since v_old IS V^T from SVD.
            V is orthogonal, so V^{-1} = V^T, making the round-trip exact.

        Component 2 — Weight regularization (Eq 9/13):
            Adds a gradient penalty that pulls weights toward theta_old.
            Equivalent to adding L_reg = (beta/2) * ||theta - theta_old||^2 to the loss,
            which gives gradient: beta * (theta - theta_old).

            Note on the paper's notation: Eq 13 writes "theta' = theta + grad* + beta * theta_old".
            Taken literally, this adds old weights directly, causing unbounded weight growth.
            The standard continual learning interpretation (consistent with EWC, SI, MAS)
            is a regularization pull toward theta_old, which is what we implement here.
            The beta hyperparameter can be tuned to achieve the desired regularization strength.

        Args:
            loss: the computed loss value (will call .backward() on it)
            beta: hyperparameter controlling strength of weight regularization toward theta_old
        """
        # Compute raw gradients for all parameters
        loss.backward()

        has_v_old = self.v_old is not None
        has_theta_old = len(self.theta_old) > 0

        if not has_v_old and not has_theta_old:
            return

        for name, param in self.named_parameters():
            # Skip params with no gradient (e.g., frozen layers, unused params)
            if param.grad is None:
                continue

            if name not in self._encoder_decoder_params:
                continue

            # --- Component 1: V-projection of gradients (Eq 10-12) ---
            # Only applies to parameters whose last dimension = embed_dim,
            # since V^T operates in the E-dimensional channel eigenspace.
            if has_v_old and param.grad.shape[-1] == self.embed_dim:
                # Eq 10: Project gradient into channel eigenspace.
                # v_old IS V^T from the paper (torch.linalg.svd returns Vh = V^T).
                # For params of shape (..., E): right-multiply by V = v_old^T
                #   grad_eigen = grad @ V = grad @ v_old.T
                grad_eigen = param.grad @ self.v_old.T  # (..., E) @ (E, E) = (..., E)

                # Eq 11-12: Scale by Omega(k, c) = k * log(c).
                grad_scaled = self.omega * grad_eigen  # (..., E) * (E,) = (..., E)

                # Eq 12: Project back to original parameter space.
                # Since V is orthogonal: V^{-1} = V^T = v_old.
                #   grad_star = grad_scaled @ V^T = grad_scaled @ v_old
                param.grad.data = grad_scaled @ self.v_old  # (..., E) @ (E, E) = (..., E)

            # --- Component 2: Weight regularization toward theta_old (Eq 9/13) ---
            if has_theta_old and name in self.theta_old:
                param.grad.data = param.grad.data + beta * (param.data - self.theta_old[name])

        return

    def train_one_epoch(self, dataloader, optimizer, task_num, **kwargs):
        """
        Train a model on one epoch.
        Args:
            dataloader: Dataloader for the training data.
            optimizer: torch.optim.Optimizer
            task_num: Which task is being trained on, starting at 1
            kwargs: Additional keyword arguments

        Returns: epoch_loss, the total accumulated loss for that epoch

        Note: After finishing ALL epochs for a given task, the caller must
        call self.save_task_state() before starting the next task.
        """
        self.train()

        epoch_loss = 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            imgs = data['image'].to(self.device)
            x_recon, d_out, s_vals = self.forward(imgs)
            batch_size = imgs.shape[0]
            loss = IUF_Loss(x=imgs,
                            x_recon=x_recon,
                            singular_vals=s_vals,
                            t=1,
                            discrim_output=d_out,
                            task_idx=(torch.ones(batch_size, dtype=torch.long) * (task_num-1)).to(self.device)
                            )
            self.update_grad(loss)
            epoch_loss += loss.item()
            optimizer.step()

        return epoch_loss

    def eval_one_epoch(self, dataloader):
        """
        Evaluate model on one epoch, collecting raw model outputs for later analysis.

        This method performs inference on the entire dataset and returns the raw
        reconstruction outputs, original images, and ground truth masks. These can
        then be used by separate prediction methods for image-level or pixel-level
        anomaly detection.

        Args:
            dataloader: Dataloader for the evaluation data

        Returns:
            x_recon_all: Tensor of reconstructed images, shape (D, C, H, W)
            x_true_all: Tensor of original images, shape (D, C, H, W)
            gt_masks_all: Tensor of ground truth masks, shape (D, C, H, W)

        Where:
            D = total number of samples in the dataset
            C = number of channels (3 for RGB)
            H, W = image height and width (224, 224)
        """
        self.eval()

        x_recon_list = []
        x_true_list = []
        gt_masks_list = []
        labels = []

        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch_idx, data in enumerate(dataloader):
                # Move images to device and perform forward pass
                imgs = data['image'].to(self.device)
                x_recon, discrim_out, singular_vals = self.forward(imgs)

                # Collect reconstructed and original images (move to CPU to save GPU memory)
                x_recon_list.append(x_recon.cpu())
                x_true_list.append(imgs.cpu())

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

        # Concatenate all batches into single tensors of shape (D, C, H, W)
        x_recon_all = torch.cat(x_recon_list, dim=0)
        x_true_all = torch.cat(x_true_list, dim=0)
        gt_masks_all = torch.cat(gt_masks_list, dim=0)  # legacy shape
        labels_all = torch.tensor(labels)

        # Return gt_masks_list (the raw per-image list) too, used by calc_results
        # to recover binary GT masks via stack_gt_masks() for pixel-AUROC.
        return x_recon_all, x_true_all, gt_masks_all, labels_all, gt_masks_list

    def calc_recon_error(self, dataloader, return_gt_masks_raw: bool = False):
        """
        Calculate total reconstruction error for one dataloader.

        Args:
            dataloader: DataLoader for the dataset.
            return_gt_masks_raw: if True, also returns the raw GT mask list for
                                 pixel-level metric computation.

        Returns:
            (recon_error, labels) by default, or
            (recon_error, labels, gt_masks_raw_list) if return_gt_masks_raw=True.
        """
        # all tensors below have shape (N, C, H, W), where N=dataset size
        x_recon_all, x_true_all, gt_masks_all, labels, gt_masks_raw = \
            self.eval_one_epoch(dataloader)
        recon_error = (x_true_all - x_recon_all).abs()
        # Both are of size (N)
        # Shapes (N, C, H, W) and (N)
        # recon_error is in [0, 1] and labels predicts whether image has anomaly (1) or not (0)
        if return_gt_masks_raw:
            return recon_error, labels, gt_masks_raw
        return recon_error, labels

    def calc_percentiles(self, train_dataloader, percentile=0.975):
        """
        For testing, evaluates thresholds for reconstruction error of normal samples
        """
        recon_error, labels = self.calc_recon_error(train_dataloader)

        # (N)
        total_recon_error = torch.tensor([recon_error[i].sum() for i in range(len(recon_error))])

        # Returns one float, the threshold at which 95% (percentile) amount of normal
        # samples are below in recon_error
        return torch.quantile(total_recon_error, percentile).item()

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
        # If pixel_auroc is requested, also retrieve the raw GT mask list for binarization.
        need_pixel = "pixel_auroc" in metrics
        if need_pixel:
            recon_error, labels, gt_masks_raw = self.calc_recon_error(
                dataloader, return_gt_masks_raw=True
            )
        else:
            recon_error, labels = self.calc_recon_error(dataloader)
        train_dataloader = kwargs.get("train_dataloader")
        threshold = self.calc_percentiles(train_dataloader)
        total_recon_error = torch.tensor([recon_error[i].sum() for i in range(len(recon_error))])
        preds = (total_recon_error > threshold).int()

        # Pre-compute pixel-level scores at 224x224 (matches GT mask resolution).
        # IUF reconstruction error is (N, C, H, W); collapse channel dimension by mean.
        pixel_scores_224 = None
        gt_masks_binary_224 = None
        if need_pixel:
            pixel_scores_224 = recon_error.mean(dim=1).float().cpu()  # (N, 224, 224)
            gt_masks_binary_224 = stack_gt_masks(gt_masks_raw, target_size=224)

        # Go through metrics for current dataset and experiment
        result_files = os.listdir("results/eval_metrics") # assumes this is run from the root folder src
        for m in metrics:
            filename = f"MTD_{task.split('_')[0]}_{exp}_{m}.csv" if dataset == "MTD" else f"{dataset}_{exp}_{m}.csv"
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
            model_name = "IUF_final" if final else "IUF"
            if m == "img_acc":
                acc = ((preds==labels).sum() / len(preds)).item()
                df.loc[model_name, task] = acc
            elif m == "img_recall":
                recall = (((preds==1)*(labels==1)).sum() / ((labels==1).sum())).item()
                df.loc[model_name, task] = recall
            elif m == "img_auroc":
                df.loc[model_name, task] = tef.binary_auroc(total_recon_error, labels).item()
            elif m == "pixel_auroc":
                df.loc[model_name, task] = compute_pixel_auroc(
                    pixel_scores_224, gt_masks_binary_224
                )

            # Update DF and save it
            df.to_csv(os.path.join("results/eval_metrics", filename))

        return

    def forward(self, x):
        x = x.to(self.device)

        # oasa_features = list of length num_layers,
        # where each item is a tensor of size (B x L x E)

        # d_out is of size (B x num_classes)
        oasa_features, d_out = self.discriminator(x, return_features=True)

        z, u, s, v = self.encoder(x, oasa_features)
        # z = latent features, (B x L x E)
        # u, s, v from SVD
        # u, (B x B)
        # S, (B | C), whichever is smaller
        # V, (C x C), C = channels/embed_dim

        x_recon = self.decoder(z)

        # =====================================================================
        # [FIX #1] Store V from this forward pass in a temporary variable.
        # save_task_state() will read _current_v at the END of the task to
        # populate v_old for the next task.
        #
        # OLD BUG: self.v_old = v.clone() was set here, meaning v_old always
        # came from the CURRENT batch during training, not the previous TASK.
        # The whole point of v_old is to protect the previous task's channel
        # space — using the current batch's V defeats the entire purpose.
        # =====================================================================
        self._current_v = v.clone().detach()

        return x_recon, d_out, s

    def save(self, model_path):
        """
        Saves the model to disk
        Args:
            model_path: a string of the model path
        Returns:
        """
        # Save model dict, based on BaseAnomalyDetector
        super().save(path=model_path)

        return

    def load(self, model_path):
        """
        Loads a model from a file
        Args:
            model_path: a string of the model path

        Returns:

        """
        # Load model dict, based on BaseAnomalyDetector
        super().load(path=model_path)

        return

def IUF_Loss(x, x_recon, singular_vals, t, discrim_output, task_idx,
             lamdba1=1.0, lambda2=0.5, lambda3=8):
    """
    Loss function consists of the following components:
        - Reconstruction error = abs(x_recon - x)
        - Discriminator error = CrossEntropy(discrim_output, true label)
        - Singular Value error = sum(singular_vals from t -> C), from SVD(M_hat),
                                where t is a hyperparameter and C is the total number of singular values.
        Each component is weighted by a corresponding lambda, where by default from the paper
        - lambda1 = 1
        - lambda2 = 0.5
        - lambda3 = 1-10

    Args:
        x: true img
        x_recon: reconstructed img
        singular_vals: Torch.Tensor(B | C), singular values from SVD of latent space
        t: int, a hyperparameter to control how many of the singular values are summed up,
                sum(singular_vals[t:len(singular_vals)]),
                where t=0 indicates that all singular values are regulated. t is equal
                to the number of top singular values we want to exclude from regularization
        discrim_output: Torch.Tensor(B, num_classes), output of discriminator
        task_idx: Torch.Tensor(B), true task indices
        lamdba1, lambda2, lambda3: weight hyperparams for 3 loss components

    Returns: Final IUF loss, based on given lambda parameters
    """
    assert t <= len(singular_vals), "t should be less than or equal to the number of singular values"

    ### Reconstruction Error
    recon_error = torch.abs(x_recon - x).mean()
    # print("Reconstruction error: ", recon_error.item())

    ### Discriminator Error
    d_err = F.cross_entropy(discrim_output, task_idx, reduction='mean')
    # print("Discriminator error: ", d_err.item())

    ### Singular Value Error
    s_val_error = singular_vals[t:].sum()
    # print("Singular Value error: ", s_val_error.item())

    return (lamdba1 * recon_error) + \
        (lambda2 * d_err) + \
        (lambda3 * s_val_error)
