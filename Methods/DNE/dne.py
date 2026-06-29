"""
Going through the paper, we want to make sure our implementation matches theirs.

Algorithm Notes:

    Biggest issue is that CutPaste is used on all experiments. Without that, model performs poorly.
    
    TRAINING
    - The embeddings I feel good with, where a batch of image samples, I with dimension B x C x H x W
        will get transformed to a vector of size B x D, where D is the dimension of the embeddings (768).
    - z_epoch should be a list the length of the dataset, where each item is a vector of size D = 768, 
        one vector for each image, based on their training loop description 
    - Regardless, the paper shows that the normal distribution characteristics are calculated based on
        each individiaul z, with a dimension of D. They do specifically mention this is only done on the last epoch
        (epoch 50), right after eq. 5. So, the mean is a vector of size D, and covariance matrix should be of size DxD.
    - the head classifier is frozen after the first task, but PyTorch's CrossEntropy doesn't need softmax output
    
    TESTING
    - Given the memory mechanism, where every task has a N, mean, cov, N samples are taken from each task distribution
        and used to create a global distribution (almost re-creating the dataset, essentially). 
    - We then find global distribution parameters of mean, cov_shrunk.
    - Mahalanobis distance is compared with those global distribution values and the embeddings of the new sample.
        They kind of treat their method as unsupervised without actually mentioning it, and the Mahalanobis distance
        should have some kind of threshold to distinguish when a sample is far enough away, it means that that 
        sample is an anomaly.   
     
    EXPERIMENTS
    - It seems that during training, one task is trained at a time, then after training on task t, testing is done
        to find the accuracy on all previous samples from tasks 1 to t. 
    - So, the accuracy reported is done using the Mahalanobis distance. After training on task t,
        all tasks 1-t are sampled to create a distribution for those tasks 1-t. Testing could be done on all
        test samples from tasks 1 to t, or invidually per task and averaged.  
    - They do mention a DNE+DER (Dark Experience Replay) method with slight results improving (~ +3-4% or so), 
        but they don't mention which images are saved and when this replay happens, so we will skip this for now.
    - Need to adjust self.train_one_epoch() to save data given path variables
"""

import pandas as pd
import os
from Methods import *
from Methods import BaseAnomalyDetector
import torcheval.metrics.functional as tef

# Distribution of Normal Embeddings
class DNE_Model(BaseAnomalyDetector):
    def __init__(self):
        super().__init__()  # Make sure to inherit all methods/properties from BaseAnomalyDetector

        # Get pre-trained Vision Transformer (ViT-B_16, like paper)
        self.vit = vit_b_16(weights='DEFAULT')  # Pre-trained weights on ImageNet
        # Freeze patch embedding layer, based on code implementation
        for param in self.vit.conv_proj.parameters():
            param.requires_grad = False
        # head is its own module, so we can easily freeze layers in autograd
        self.head = nn.Sequential(
            nn.Linear(in_features=768, out_features=2)
        )
        # Creating memory mechanism
        # Total memory, M, which will contain a list of statistics (mean, cov, num) for each task
        self.memory = []
        # z_epoch holds embeddings for samples in  last epoch (epoch 50 in paper)
        self.z_epoch = None  # num_samples x 768
        # Global distribution params for calculating mahalanobis distance
        self.global_mu = None
        self.global_cov = None
        # move to best device
        self.to(self.device)
        return

    def forward(self, img, head=False, add_to_z_epoch=False):
        # Calculate embeddings, Z, from ViT
        embeds = self.embed(img)  # B x 768

        # Add embeddings to z_epoch if in training
        if add_to_z_epoch:
            with torch.no_grad():
                z = embeds.detach().clone()
                if self.z_epoch is None:
                    self.z_epoch = z
                else:
                    self.z_epoch = torch.cat((self.z_epoch, z))

        # Return logits or embeddings, based on the head param
        if head:
            logits = self.head(embeds)
            return logits

        return embeds

    def embed(self, img):
        """
        Gets the final feature embeddings of the ViT-B_16 model, before passing to the head for classification.
        Each of these layers was checked against the original ViT paper, code implementation, and current paper
        - Takes in any image of whatever size and does transformation
        - Outputs the final class-token, which is of size (batch_size B, 768 (Latent dimension size=768 per token/patch))
        """
        # Creates patch_embeddings based on transformed image
        img = img.unsqueeze(0) if len(
            img.shape) == 3 else img  # Make sure we have a batch dimension if single, unbatched image was given
        patch_embeds = self.vit._process_input(img)  # B x 196 x 768 (196 patches/tokens, latent dimension D = 768)
        # Expand class token to the full batch, size B x 1 x 768
        batch_class_token = self.vit.class_token.expand(img.shape[0], -1, -1)
        # Add class token to patch embeddings
        patch_embeds = torch.cat([batch_class_token, patch_embeds], dim=1)  # B x 197 x 768
        # Pass the patch embeddings through the transformer encoder
        features = self.vit.encoder(patch_embeds)  # B x 197 x 768
        # We only want the final output features of the class token, so a 1D vector of size D (768) for each img
        cls_features = features[:, 0]  # B x 768
        return cls_features

    def freeze_head(self, freeze):
        # freeze is either True/False, whether to freeze/unfreeze the head layer module parameters (won't update)
        for param in self.head.parameters():
            param.requires_grad = not freeze
        return

    def update_memory(self, **kwargs):
        """
        Updates the long-term memory, M, based on z_epoch,
        which after epoch 50 should be added into memory
        z_epoch should be of size (num_task, 768), where
            num_task is the number of samples in the current task
            768 is the ViT latent dimension size, D
        """
        if self.z_epoch is None:
            raise Exception("z_epoch not created")
        else:
            task_memory = []  # will contain num_task, mean, covariance
            # Add num_task
            task_memory.append(self.z_epoch.shape[0])
            # Add mean vector, should be a 1D tensor of length 768
            task_memory.append(self.z_epoch.mean(dim=0))
            # Add shrunk covariance, based on equations 4 and 6 from paper
            # Should be a 2D vector of size D x D (D = 768)
            task_memory.append(self._calc_shrunk_cov(self.z_epoch))

            self.z_epoch = None
            self.memory.append(task_memory)
            return

    def _calc_shrunk_cov(self, data, alpha=0.5):
        """
        Calculates the shrunk covariance matrix
        Parameters:
            data (Tensor): Tensor containing the data, of size N x D
            alpha (float): alpha parameter, in [0, 1] (can't find mention in paper of what alpha should be)
        Returns: shrunk covariance
        """
        cov = torch.cov(data.T).cpu() # of size D x D
        diag = torch.diag(cov) # of size D
        D = diag.shape[0]
        trace = diag.sum().item()
        coeff = (alpha * trace) / D

        return ((1 - alpha) * cov) + (coeff * torch.eye(D))

    def generate_global_dist(self):
        """
        Based on the paper, a global mean and covariance distribution sampling is generated from the memory.
        We have to generate the samples, because the sklearn ShrunkCovariance class has to be used on cpu,
            which may conflict with model being on GPU or MPS.
        """
        # we want to re-generate our original dataset from the task distributions in memory
        self.global_mu = None
        self.global_cov = None
        z_global = None
        for t in self.memory:
            n_t, mu, shrunk_cov = t
            gaussian = Normal(mu.cpu(), shrunk_cov.diag().cpu())
            task_samples = gaussian.sample((n_t,))  # samples n_t samples from task guassian
            if z_global is None:
                z_global = task_samples
            else:
                z_global = torch.cat((z_global, task_samples))
        # z_global is then a size of N x 768, where N is the size of all tasks combined
        # where each task has size n, and the sum of all n-values is N

        # Based off z_global, we will calculate a global mean/covariance for use in Mahalanobis
        self.global_mu = z_global.mean(dim=0).to(self.device) # 1D vector of size 768
        self.global_cov = self._calc_shrunk_cov(z_global).to(self.device) # 2D vector of size 768 x 768 (D x D)

        return

    def train_one_epoch(self, dataloader, optimizer, criterion,
                        task_num, **kwargs):
        """
        Train a model on one epoch.
        Args:
            dataloader: Dataloader for the training data.
            optimizer: torch.optim.Optimizer
            criterion: loss function
            task_num: Which task is being trained on, starting at 1
            kwargs: Additional keyword arguments,
                - update_z_epoch: Whether to add the global mean and covariance distribution to the memory

        Returns: epoch_loss, the total accumulated loss for that epoch

        """
        self.train()
        self.z_epoch = None
        # Only update z-epoch at final epoch
        self.freeze_head(False if task_num == 1 else True)

        epoch_loss = 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            imgs = data['image'].to(self.device)
            outputs = self.forward(imgs, head=True,
                            add_to_z_epoch=kwargs.get("update_z_epoch")).cpu()
            labels = torch.tensor(data['label'])
            loss = criterion(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        return epoch_loss

    def eval_one_epoch(self, dataloader):
        """
        Eval a model on one epoch.
        Args:
            dataloader: Dataloader for the training data.
            results_path: If exists, where we want to save data about the results
            model_path: If exists, where we want to access model params

        Returns:
            predictions: the predicted labels
            labels: the ground truth labels
        """
        self.eval()

        all_labels = []
        scores = []
        with torch.no_grad():
            for batch_idx, data in enumerate(dataloader):
                imgs = data['image'].to(self.device)
                for img in imgs:
                    scores.append(self.predict(img))
                all_labels += data['label']

        # Note: each of these values are just float values (taken from Tensor.item())
        return scores, all_labels

    def calc_percentiles(self, train_dataloader, percentile=0.975):
        """
        For testing, evaluates thresholds for reconstruction error of normal samples
        """
        scores, labels = self.eval_one_epoch(train_dataloader)
        scores = torch.tensor(scores) # N
        labels = torch.tensor(labels) # N
        normal_scores = scores[labels == 0]

        # Returns one float, the threshold at which 95% (percentile) amount of normal
        return torch.quantile(normal_scores, percentile).item()

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
        Returns:
            Nothing; saves df's to appropriate files
        """

        scores, labels = self.eval_one_epoch(dataloader)
        scores = torch.tensor(scores)
        labels = torch.tensor(labels)
        threshold = self.calc_percentiles(kwargs.get("train_dataloader"))
        preds = (scores > threshold).int()

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
            model_name = "DNE_final" if final else "DNE"
            if m == "img_acc":
                acc = ((preds==labels).sum() / len(preds)).item()
                df.loc[model_name, task] = acc
            elif m == "img_recall":
                recall = (((preds==1)*(labels==1)).sum() / ((labels==1).sum())).item()
                df.loc[model_name, task] = recall
            elif m == "img_auroc":
                df.loc[model_name, task] = tef.binary_auroc(scores, labels).item()
            elif m == "pixel_auroc":
                # DNE produces only an image-level Mahalanobis distance from the
                # CLS-style 768-D embedding (no spatial information). Pixel-level
                # AUROC is not well-defined for this method; we record NaN so the
                # rebuttal table can render the entry as ``---'' instead of a
                # spurious 0.5.
                df.loc[model_name, task] = float('nan')

            # Update DF and save it
            df.to_csv(os.path.join("results/eval_metrics", filename))

        return

    def predict(self, img):
        """
        Calculates the Mahalanobis distance for a given image as our prediction
        Args:
            img: should be an image read in as a tensor of size (3, 224, 224)

        Returns: Mahalanobis distance, a number
        """
        # Embedding
        img = img.to(self.device)
        z = self.forward(img)

        # Make sure to generate
        if self.global_mu is None or self.global_cov is None:
            self.generate_global_dist()

        ### Calculate the Mahalanobis Distance and return it
        # z - z_bar
        diff = (z - self.global_mu.to(self.device)).detach().clone() # 1D vector of size 1 x 768
        # Shrunk covariance inverse — computed on CPU to avoid cuSOLVER dependency
        # (768x768 inverse is trivial on CPU, avoids libtorch_cuda_linalg issues on Jetson)
        inv = torch.linalg.inv(self.global_cov.cpu()).to(self.device).detach().clone() # 768 x 768

        # Start calculating distance
        dist = torch.matmul(diff, inv) # output is 1 x 768
        dist = torch.matmul(dist, diff.T) # output is 1 x 1

        return dist.sqrt()[0][0].item()

    def save(self, model_path):
        """
        Saves the model to disk, as well as memory
        Args:
            model_path: a string of the model path
        Returns:
        """
        # Save model dict, based on BaseAnomalyDetector
        super().save(path=model_path)

        # Find memory path and save memory
        memory_path = model_path.split("weights")[0] + "memory.pth"
        # Assumes that self.update_memory() has been called
        torch.save(self.memory, memory_path)

        return

    def load(self, model_path):
        """
        Loads a model from a file, as well as memory
        Args:
            model_path: a string of the model path

        Returns:

        """
        # Load model dict, based on BaseAnomalyDetector
        super().load(path=model_path)

        # Find memory path and load that as well
        memory_path = model_path.split("weights")[0] + "memory.pth"
        self.memory = torch.load(memory_path, map_location=self.device)

        return



