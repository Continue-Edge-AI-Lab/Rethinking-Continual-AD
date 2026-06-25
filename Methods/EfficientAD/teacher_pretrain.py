### Teacher Training, following Algorithm 3 from the paper

import os
from einops import rearrange
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from .image_net import ImageNetDataset
from .patch_description_network import PDN
from .resnet import ResNet

CALC_NORM_PARAMS = True
TRAIN_TEACHER = True

# pdn input = torch.tensor of size (B, 3, 256, 256)
pdn = PDN('s', student=False)
# resnet/phi input = torch.tensor of size (B, 3, 512, 512)
resnet = ResNet('resnet34', efficient_ad=True)
# ResNet requires gradients to be turned off
for param in resnet.parameters():
    param.requires_grad = False
# Both models output tensor of size (B, 384, 64, 64)

# Compute feature extractor channel normalization params
mean_transform = T.Compose([
    T.Resize((512, 512)),
    T.RandomGrayscale(p=0.1),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = ImageNetDataset(transform=mean_transform)

teacher_mean = None
teacher_std = None

if CALC_NORM_PARAMS:
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print('Starting to calculate channel mean and std...')

    # Values for Welford's algorithm of calculating running mean/std
    count = 0
    mu = None
    m2 = None

    for i in range(10_000):
        if i % 1000 == 0:
            print(f"iteration {i}")

        imgs = next(iter(dataloader))
        features = resnet(imgs) # (1, 384, 64, 64)
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

    teacher_mean = mu
    teacher_std = torch.sqrt(m2 / (count - 1))

    torch.save(teacher_mean, "phi_mean.pth")
    torch.save(teacher_std, "phi_std.pth")
    print('Finished calculating channel mean and std...')

# Do training for teacher
if TRAIN_TEACHER:
    optimizer = torch.optim.Adam(pdn.parameters(), lr=0.001, weight_decay=.0001)
    loss_fn = nn.MSELoss()
    dataset.transform = None
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    phi_transform = T.Compose([
        T.Resize((512, 512)),
        T.RandomGrayscale(p=0.1),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    teacher_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomGrayscale(p=0.1),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # We need to load in the params now, for training. Each has size 384
    if teacher_mean is None:
        teacher_mean = torch.load("phi_mean.pth")
    if teacher_std is None:
        teacher_std = torch.load("phi_std.pth")

    training_loss = []
    for i in range(60_000):

        # Have to get one image at a time because of the different transforms
        phi_imgs = None
        for b in range(16):
            new_img = next(iter(dataloader))
            # Get features from phi (ResNet feature extractor)
            new_phi_img = phi_transform(new_img) # (1, 3, 512, 512)
            # Get features from Teacher PDN network
            new_teacher_img = teacher_transform(new_img) # (1, 3, 256, 256)
            if phi_imgs is None:
                phi_imgs = new_phi_img
                teacher_imgs = new_teacher_img
            else:
                phi_imgs = torch.cat((phi_imgs, new_phi_img), dim=0)
                teacher_imgs = torch.cat((teacher_imgs, new_teacher_img), dim=0)

        # phi_imgs now has shape (16, 3, 512, 512) for our batch
        # teacher_imgs now has shape (16, 3, 256, 256) for our batch
        phi_features = resnet(phi_imgs)
        phi_features = rearrange(phi_features, "b c h w -> b h w c")
        phi_features = (phi_features - teacher_mean) / teacher_std # (16, 64, 64, 384)
        phi_features = rearrange(phi_features, "b h w c -> b c h w") # (16, 384, 64, 64)

        teacher_features = pdn(teacher_imgs) # (16, 384, 64, 64)

        # Calculate loss and do backprop
        optimizer.zero_grad()
        loss = loss_fn(phi_features, teacher_features)
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())

        if i % 1000 == 0:
            print(f"Iteration {i}")
            print(f"loss: {loss.item():.7f}")
            print("---------------")

    torch.save(pdn.state_dict(), "teacher.pth")