import os

import torch.mps
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import torchvision.io

class ImageNetDataset(Dataset):
    """
    The downloaded ImageNet 2017 dataset. User needs to download this and

    NOTES:
        - images are output as torch.Tensors with size (C, H, W)
        - Has 3 categories: train, test, val
        - The test set doesn't have annotations
        - In this implementation, we just need the images for feature extraction
    """
    def __init__(self,
                 img_dir: str = "C:/Users/chadw/Documents/ImageNet_2017/Data/CLS-LOC",
                 set: str = "train",   # either 'train', 'val', or 'test'
                 transform=None
                 ):
        self.set = set
        self.path = img_dir
        self.img_path = f"{self.path}/{self.set}"
        # Set transform
        self.transform = transform
        self.to_pil = T.ToPILImage()

        # Get all filenames (prefixes, without file extension)
        self.filenames = []
        self._get_all_filenames()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        return

    def _get_all_filenames(self):
        # all images are just in the main folder
        if self.set in ["val", "test"]:
            for file in os.listdir(self.img_path):
                if file.endswith((".jpg", ".png", ".jpeg", ".JPG", ".JPEG")):
                    self.filenames.append(file)
        # For the training set, there are folders for each image class
        else:
            for folder in os.listdir(self.img_path):
                for file in os.listdir(self.img_path + "/" + folder):
                    if file.endswith((".jpg", ".png", ".jpeg", ".JPG", ".JPEG")):
                        self.filenames.append(folder + "/" + file)
        return

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get images
        img = torchvision.io.read_image(f"{self.img_path}/{self.filenames[idx]}",
                                        apply_exif_orientation=True,
                                        mode=torchvision.io.ImageReadMode.RGB).to(self.device)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def get_pil(self, idx):
        return self.to_pil(self.__getitem__(idx))





