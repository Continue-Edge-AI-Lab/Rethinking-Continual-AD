import torch
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torcheval.metrics.functional as tef
from Methods import (
    BaseAnomalyDetector,
    compute_pixel_auroc,
    upsample_anomaly_map,
    stack_gt_masks,
)
from .image_net import ImageNetDataset
from .patch_description_network import PDN
from .autoencoder import Autoencoder
from einops import rearrange


# noinspection PyCallingNonCallable,PyTypeChecker
class EfficientAD_Model(BaseAnomalyDetector):
    """
    Method to perform anomaly detection using EfficientAD, described as follows:

    TRAINING
    1. For pre-training, the teacher PDN is first trained to mimic a ResNet (or other feature extractor)
        in producing the same features from ImageNet. See teacher_pretraining.py for more details.
    2. There are 3 different loss functions, each uses only normal samples:
        i. L_ST = The student's first 368 vectors are trained to match the teacher,
            and a random Imagenet image's penalty.

            The student only is trained on the 0.999 percentile of channel differences (MSE differences)

            The penalty is designed to constrain the student from learning anomalous features

        ii. L_AE = The autoencoder is trained to match the teacher's output

            Since it has a small latent space, the idea is that it only finds logical/global anomalies

        iii. L_STAE = The student's second 368 vectors are trained to match the autoencoder's output

    INFERENCE
    1. Given a new sample,
    """
    def __init__(self,
                 teacher_size: str = 's',
                 student_size: str = 's',
                 p_hard: float = 0.999,
                 edge_testing = False
                 ):

        # Get device and move all modules there
        super().__init__()

        ### Our modules
        self.teacher = PDN(teacher_size, student=False)
        # Assumes the teacher is already trained and teacher.pth is in the EfficientAD folder
        if edge_testing:
            teacher_state_dict = torch.load("../Methods/EfficientAD/teacher.pth", map_location=self.device)
        else:
            teacher_state_dict = torch.load("Methods/EfficientAD/teacher.pth", map_location=self.device)
        self.teacher.load_state_dict(teacher_state_dict)
        self.teacher.to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher_mean = None
        self.teacher_std = None


        self.student = PDN(student_size, student=True).to(self.device)
        self.autoencoder = Autoencoder().to(self.device)
        # Transform given to datasets
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]).to(self.device)
        # Upsamppling for anomaly map
        self.anomaly_map_upsample = T.Resize((256, 256)).to(self.device)
        # p_hard quantile used to calculate d_hard
        self.p_hard = p_hard

        return

    def calc_teacher_params(self, task_dataset):
        """
        Calculates the teacher mean and standard deviation for the given task dataset.
        This is used to normalize the feature outputs during training and inference.
        """
        # Values for Welford's algorithm of calculating running mean/std
        count = 0
        mu = None
        m2 = None

        with torch.no_grad():
            for idx, data in enumerate(task_dataset):
                img = data['image'].unsqueeze(0).to(self.device) # (1, 3, 256, 256)
                features = self.teacher(img) # (1, 384, 64, 64)
                features = rearrange(features, "b c h w -> (b h w) c") # (4096, 384)

                # Standard Welford's algorithm, one sample at a time
                for sample in features: # sample is (384,)
                    count += 1
                    if mu is None:
                        mu = sample.clone()
                        m2 = torch.zeros_like(sample)
                    else:
                        delta = sample - mu
                        mu = mu + delta / count
                        delta2 = sample - mu  # mu has changed
                        m2 = m2 + delta * delta2

            self.teacher_mean = mu
            self.teacher_std = torch.sqrt(m2 / (count - 1))

        return


    def forward(self, img: torch.Tensor):
        """
        Passes batched images and gets final features for inference or training.
        Assumes that calc_teacher_params() has already been run
        :param img: batched images of size (B, 3, 256, 256)
        :return:
        """
        # Get normalized teacher features
        t_features = self.teacher(img) # (B, 384, 64, 64)
        t_features = rearrange(t_features, "b c h w -> b h w c")
        t_features = (t_features - self.teacher_mean) / self.teacher_std
        t_features = rearrange(t_features, "b h w c -> b c h w") # (B, 384, 64, 64)

        # Get student features
        s_features = self.student(img) # (B, 768, 64, 64)

        # Get AutoEncoder features
        ae_features = self.autoencoder(img) # (B, 384, 64, 64)

        return (t_features,     # (B, 384, 64, 64)
                s_features,     # (B, 768, 64, 64)
                ae_features)    # (B, 384, 64, 64)

    def calc_st_loss(self, t_features, s_features):
        """
        Given the computed teacher features and the first 384 features of the student,
        we want to calculate the student loss (L_st in the paper).

        We ignore the ImageNet penalty, as the increased cost of training isn't worth it
        for ~ 0.004 AUROC improvement

        Both feature sets are of size (B, 384, 64, 64)
        """
        dists = F.mse_loss(t_features, s_features, reduction='none') # (1, 384, 64, 64)
        dists = rearrange(dists, "b c h w -> (b h w c)") # (N)
        d_hard = torch.quantile(dists, self.p_hard)
        dists = dists[dists >= d_hard]
        return dists.mean()

    def train_one_epoch(self, optimizer, **kwargs):
        """
        Train one epoch on the given task dataloader. We will ignore the map normalization,
        as we don't have a validation set and it only improves the model ~ 0.005 on AUROC.

        We also don't perform augmentation, as our other methods do not benefit from augmentation,
        and we want to compare models fairly.

        Finally, we did add a parameter to the st_loss to limit the loss and help the model actually learn
        """
        dataset = kwargs.get('dataset')

        epoch_loss = 0
        for idx, data in enumerate(dataset):
            # Get initial features
            img = data['image'].unsqueeze(0).to(self.device) # (1, 3, 256, 256)
            t_features, s_features, ae_features = self.forward(img)

            st_loss = self.calc_st_loss(t_features, s_features[:, 0:384, :, :])
            ae_loss = F.mse_loss(t_features, ae_features, reduction='mean')
            stae_loss = F.mse_loss(ae_features, s_features[:, 384:768, :, :], reduction='mean')

            total_loss = st_loss + ae_loss + stae_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        return epoch_loss

    def eval_one_epoch(self, dataloader, return_pixel_maps: bool = False, **kwargs):
        """
        Evaluate one epoch on the given task dataloader. We will ignore the map normalization,
        as we don't have a validation set and it only improves the model ~ 0.005 on AUROC.

        We also don't perform augmentation, as our other methods do not benefit from augmentation,
        and we want to compare models fairly.

        Args:
            dataloader: DataLoader.
            return_pixel_maps: if True, additionally returns (anomaly_maps_full,
                gt_masks_raw_list) for pixel-AUROC. Default False keeps the
                legacy 3-tuple return signature so existing callers (calc_percentiles)
                are unaffected.

        Returns:
            (ad_scores, gt_masks, labels) by default, or
            (ad_scores, gt_masks, labels, anomaly_maps_full, gt_masks_raw_list)
            when return_pixel_maps=True. anomaly_maps_full has shape (N, 256, 256).
        """
        self.eval()

        img_ad_scores = []
        gt_masks_list = []
        labels = []
        anomaly_maps_collected = []  # populated only if return_pixel_maps=True
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                # Get initial features
                imgs = data['image'].to(self.device) # (B, 3, 256, 256)
                t_features, s_features, ae_features = self.forward(imgs)

                # All have shape (B, 64, 64)
                m_st = F.mse_loss(t_features, s_features[:, 0:384, :, :], reduction='none').mean(dim=1)
                m_stae = F.mse_loss(ae_features, s_features[:, 384:768, :, :], reduction='none').mean(dim=1)
                anomaly_maps = (0.5*m_stae + 0.5*m_st)
                # (B, 256, 256)
                anomaly_maps = self.anomaly_map_upsample(anomaly_maps)
                if return_pixel_maps:
                    anomaly_maps_collected.append(anomaly_maps.detach().cpu())
                for map in anomaly_maps:
                    map = map.flatten()
                    img_ad_scores.append(map.max().item())

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
        gt_masks = torch.stack(gt_masks_list) # (N, 3, 256, 256)
        labels = torch.tensor(labels) # N

        if return_pixel_maps:
            anomaly_maps_full = torch.cat(anomaly_maps_collected, dim=0)  # (N, 256, 256)
            return ad_scores, gt_masks, labels, anomaly_maps_full, gt_masks_list
        return ad_scores, gt_masks, labels

    def calc_percentiles(self, train_dataloader, percentile=0.975):
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
        # If pixel_auroc is requested, collect per-pixel anomaly maps (256x256) and raw GT masks.
        need_pixel = "pixel_auroc" in metrics
        if need_pixel:
            img_ad_scores, gt_masks, labels, anomaly_maps_full, gt_masks_raw = \
                self.eval_one_epoch(dataloader, return_pixel_maps=True)
        else:
            img_ad_scores, gt_masks, labels = self.eval_one_epoch(dataloader)
        threshold = self.calc_percentiles(kwargs.get("train_dataloader"))
        preds = (img_ad_scores > threshold).int()

        # Pre-compute pixel-AUROC inputs at native EfficientAD resolution (256x256).
        pixel_scores_256 = None
        gt_masks_binary_256 = None
        if need_pixel:
            # anomaly_maps_full already at 256x256, no upsampling needed.
            pixel_scores_256 = anomaly_maps_full
            gt_masks_binary_256 = stack_gt_masks(gt_masks_raw, target_size=256)

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
            model_name = "EfficientAD_final" if final else "EfficientAD"
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
                    pixel_scores_256, gt_masks_binary_256
                )

            # Update DF and save it
            df.to_csv(os.path.join("results/eval_metrics", filename))

        return

    def save(self, path: str):
        """
        Save the following:
        - Teacher mean and standard deviation
        - Student state_dict
        - Autoencoder state_dict

        :param path: filepath to save to (without extension)
        """

        # Prepare save dictionary
        save_dict = {
            'teacher_mean': self.teacher_mean,
            'teacher_std': self.teacher_std,
            'student_state_dict': self.student.state_dict(),
            'autoencoder_state_dict': self.autoencoder.state_dict(),
        }

        # Save as .pth file
        torch.save(save_dict, path)

        return

    def load(self, path: str):
        """
        Load in params for model

        :param path: filepath to load from (without extension)
        """
        # Load the saved dictionary
        save_dict = torch.load(path, map_location=self.device)
        self.teacher_mean = save_dict['teacher_mean']
        self.teacher_std = save_dict['teacher_std']
        self.student.load_state_dict(save_dict['student_state_dict'])
        self.autoencoder.load_state_dict(save_dict['autoencoder_state_dict'])

        self.to(self.device)

        return
