"""
Our dataset classes for MVTec-LOCO

MVTEC-LOCO: https://www.mvtec.com/company/research/datasets/mvtec-loco
MVTec-LOCO contains 5 categories of images:
    - breakfast_box
    - screw_bag
    - pushpins
    - splicing_connectors
    - juice_bottle

We differentiate datasets by whether they are
- training or testing
- which task, a string of one of the categories
"""

import os
from datasets import *

class mvtec_loco(Dataset):
    def __init__(self,
                 train=True,
                 task=None,
                 replay=0,
                 transform=None):
        """
        Creates MVTEC dataset for use in PyTorch
        Args:
            train: Whether the dataset is used for training or testing, during training, only normal samples are seen
            task: Which task, a string of one of the categories
            replay: An int representing how many samples from each previous task should be included
                    in the training set. A value of zero indicates no replay
            transform: PyTorch pre-processing transforms to be applied to the images
        """
        self.train = train  # Whether this is the training or testing set
        self.task = task
        self.replay = replay
        self.unsupervised = True
        self.all_tasks = ['breakfast_box', 'screw_bag', 'pushpins',
                 'splicing_connectors', 'juice_bottle']
        # If no transform is given, we want to make sure we resize so we can batch
        # The new dims are for passing into ViT (224 x 224)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ])

        else:
            self.transform = transform

        # Path is the general path to the training/testing set of a given task
        # We need the path to also extract groud truth (gt) images
        self.path = (f"datasets/mvtec_loco/")

        self.filenames = []
        self.labels = []  # 1 = anomaly, 0 = good
        self.get_all_filenames()  # creates list of all filenames and paths, self.filenames
        return

    def get_all_filenames(self):
        """
        creates list of all image filenames, a list of strings
        """
        # Get current task images
        img_path = self.path + self.task
        img_path = (img_path + '/train') if self.train else (img_path + '/test')
        for anom_type in os.listdir(img_path):  # iterating through anomaly types
            if "." not in anom_type:  # Making sure the folder is not a file
                for img in os.listdir(f'{img_path}/{anom_type}'):  # iterating through images
                    if img.endswith(".png"):
                        self.filenames.append(f"{img_path}/{anom_type}/{img}")
                        self.labels.append(1 if anom_type != "good" else 0)
        # If using replay, then also add previous tasks
        if self.replay > 0:
            current_task_idx = self.all_tasks.index(self.task)
            for i in range(current_task_idx):
                img_path = self.path + self.all_tasks[i]
                img_path = (img_path + '/train') if self.train else (img_path + '/test')
                replay_filenames = []
                replay_labels = []
                for anom_type in os.listdir(img_path):  # iterating through anomaly types
                    if "." not in anom_type:  # Making sure the folder is not a file
                        for img in os.listdir(f'{img_path}/{anom_type}'):  # iterating through images
                            if img.endswith(".png"):
                                replay_filenames.append(f"{img_path}/{anom_type}/{img}")
                                replay_labels.append(1 if anom_type != "good" else 0)
                random_idx = random.sample(range(len(replay_filenames)), self.replay)
                sampled_replay_filenames = [replay_filenames[j] for j in random_idx]
                sampled_replay_labels = [replay_labels[j] for j in random_idx]
                self.filenames.extend(sampled_replay_filenames)
                self.labels.extend(sampled_replay_labels)

        return

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Images need to be pre-processed beforehand so dataloader handles same size images
        # need to do is make sure the image has 3 channels
        img_filename = self.filenames[idx]
        img = read_image(img_filename)
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        img = self.transform(img)

        # Get ground truth image
        img_split = img_filename.split('/')
        anom_type = img_split[-2]
        task = img_split[-4]
        if anom_type == 'good':
            gt_img = None
        else:
            img_num = img_split[-1].split('.')[0]
            gt_filename = f'{self.path}/{task}/ground_truth/{anom_type}/{img_num}/000.png'
            gt_img = read_image(gt_filename)
            gt_img = self.transform(gt_img)

        return {'image': img,
                'ground_truth_mask': gt_img,
                'label': self.labels[idx],
                'anomaly_type': anom_type}