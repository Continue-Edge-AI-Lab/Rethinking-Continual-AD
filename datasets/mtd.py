"""
This file contains the mtd dataset class, which contains only one type of object, magnetic tiles.

There are 1344 images in total, with ~70% of those images being normal.
The rest of the images are split into 5 anomalous type classes:
    - Blowhole
    - Break
    - Crack
    - Fray
    - Uneven

Each image (.jpg) has a corresponding ground truth mask file (.png).

- For supervised training, we can do a regular 70/30 split between training and testing, so that training
and testing follow similar distributions.
- For unsupervised, we can move over half of the normal images to training, but keep the rest of the images in testing,
which means we have a 35/65 training/testing split.

For MTD, we will run several sub-experiments, where there are progressive tasks. In each sub-experiment,
the dataset will be transformed at a higher and higher intensity each task, slowly shifting the data distribution
over time. This will allow us to see how well the model can adapt to the changing data distribution. Once we find
sufficient parameters for our experiments, we will save the resulting datasets for each sub-experiment.
"""

"""
TODO: 
    - After running experiments and finding good values, we want to save the dataset
"""

from datasets import *
import datasets.transforms as dt

class mtd(Dataset):
    def __init__(self,
                 train=True,
                 unsupervised=True,
                 replay=False,
                 transform=None,
                 data_aug=None,
                 data_aug_params=None,
                 ):
        """
        Creates the dataset for one task of the Magnetic Tile Defects dataset.
        Args:
            train: Whether the dataset is used for training or testing, during training, only normal samples are seen
            unsupervised: Whether to use unsupervised or supervised training, i.e. whether training uses only normal
                            samples or not
            replay: A bool representing whether samples from previous tasks should be used
            transform: PyTorch pre-processing transforms to be applied to the images
            data_aug: Data Augmentation to be applied to the images for Continual Drift. Can be a string of type:
                - 'color'
                        - adjusts color jitter
                        - simulates lighting/color/texture changes in products
                        - window values vary the pixels randomly by a factor +/- X, where X
                        is a random number in the window value. So, if X is 0.1, then the pixel will be
                        randomly adjusted by +/- 10%
                        - ALL values should range from [0, 0.5]
                - 'blur'
                        - applies spot Guassian Blur
                        - simulates sensor wear or out of focus
                        - Has two values: kernel size and sigma, where sigma is the standard deviation of the kernel.
                          Low sigma would be a small blur, while a high sigma would be a large blur
                        - kernel values should range from [3, 9], while sigma should range from [0.1, 5]
                - 'geometric', applies geometric transformations to the image to simulate product movement
                        - does a combo of slight shift, rotation, scale, and shear (changes perspective of product)
                        - simulates product placed in different ways or product deformation
                        - Each value described below is a number which describes a window
                        - degrees should vary from [0, 180],
                          translate should range from [0, 35]
                          (represents pixel values; don't want too much translation more than 10% in any direction),
                          scale should be a number between [0.0, 0.25], where the image can get scaled +/- scale
                          shear should also range from [0, 180], in the same way that rotation is
                - ...
            data_aug_params: A list of parameters for a given transformation.
                        If a given transformation can be applied on a scale of [0.0, 1.0], the params are the
                        intensity range to apply to the images. For example, noise might be applied in an intensity window
                        of [0.2, 0.3]. This would equate to one task in the continual drift setting. Each different
                        data augmentation will have different params in a list.

                        See self.set_data_aug() for more details on which parameters are needed
        """
        self.train = train
        self.unsupervised = unsupervised
        self.replay = replay

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
        self.data_aug = data_aug # Type of data augmentation
        self.data_aug_params = data_aug_params # Parameters for data augmentation
        self.continual_transform = None # The actual data augmentation to be applied

        # Path is the general path to the training/testing set of a given task
        # We need the path to also extract groud truth (gt) images
        self.path = ('datasets/magnetic_tile_defects/' +
                     f'{'unsupervised/' if unsupervised else 'supervised/'}' +
                     f'{'train' if train else 'test'}')

        self.filenames = []
        self.labels = []  # 1 = anomaly, 0 = good
        self.get_all_filenames()  # creates list of all filenames and paths, self.filenames
        self.set_data_aug()
        return

    def get_all_filenames(self):
        """
        creates list of all image filenames, a list of strings
        """
        for anom_type in os.listdir(self.path): # iterating through anomaly types
            if anom_type.startswith("MT_"): # Making sure the folder is not a file
                for img in os.listdir(f'{self.path}/{anom_type}'): # iterating through images
                    if img.endswith(".jpg"):
                        self.filenames.append(f"{self.path}/{anom_type}/{img}")
                        self.labels.append(0 if anom_type.endswith("Free") else 1)

        return

    def set_data_aug(self):
        """
        Sets the data augmentation to be applied to the images for Continual Drift.
        """
        if self.data_aug is not None:
            if self.data_aug == 'color':
                window = (self.data_aug_params[0], self.data_aug_params[1])
                self.continual_transform = dt.color_transform(window, self.replay)
            elif self.data_aug == 'blur':
                kernel_size, sigma = self.data_aug_params
                self.continual_transform = transforms.GaussianBlur(kernel_size,
                                                                   (0.5, sigma) if self.replay else sigma)
            elif self.data_aug == 'geometric':
                degrees, translate, scale, shear = self.data_aug_params
                self.continual_transform = dt.geometric_transform(degrees, translate, scale, shear, self.replay)
            pass

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Images need to be pre-processed beforehand so dataloader handles same size images
        # need to do is make sure the image has 3 channels
        img_filename = self.filenames[idx]
        img = read_image(img_filename).expand(3, -1, -1)

        # Get ground truth image
        img_split = img_filename.split('/')
        anom_type = img_split[-2]
        if anom_type == 'MT_Free':
            gt_img = None
        else:
            gt_filename = img_filename.split(".")[0] + ".png"
            gt_img = read_image(gt_filename)

        # Perform data aumentation for continual learning task, then any pre-processing to feed into model
        if self.data_aug is not None:
            img = self.continual_transform(img)
            # We want to transform the ground truth mask the same way as the image,
            # if the mask exists and the transformation alters the location of the anomaly
            if gt_img is not None and self.data_aug == "geometric":
                gt_img = self.continual_transform(gt_img)

        img = self.transform(img)
        if gt_img is not None:
            gt_img = self.transform(gt_img)

        return {'image': img,
                'ground_truth_mask': gt_img,
                'label': self.labels[idx],
                'anomaly_type': anom_type.split('_')[1]}
