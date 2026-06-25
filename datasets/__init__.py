import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.utils import make_grid
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import plotly.graph_objects as go
from PIL import Image
from .mvtec import mvtec
from .mvtec_v2 import mvtec_v2
from .mvtec_loco import mvtec_loco
from .mtd import mtd

### Collate samples appropriately
def collate(batch):
    result = {}
    for key in batch[0].keys():
        result[key] = [item[key] for item in batch]
        if key == 'image':
            result[key] = torch.stack(result[key])
    return result

### Show a list of tensor images
def show(imgs, grid_rows=None, grid_cols=None):
    """
    Displays pairs of images (image and ground truth mask) using Plotly.

    Args:
        imgs (list of torch.Tensor): List of tensors, where each pair represents an image and its mask.
        grid_rows (int, optional): Number of rows in the grid.
        grid_cols (int, optional): Number of columns in the grid.
    """

    num_pairs = len(imgs) // 2

    if grid_rows is None and grid_cols is None:
        grid_cols = int(np.ceil(np.sqrt(num_pairs)))
        grid_rows = int(np.ceil(num_pairs / grid_cols))
    elif grid_rows is None:
        grid_rows = int(np.ceil(num_pairs / grid_cols))
    elif grid_cols is None:
        grid_cols = int(np.ceil(num_pairs / grid_rows))

    fig = go.Figure()

    for i in range(num_pairs):
        img = imgs[2 * i].detach()
        mask = imgs[2 * i + 1].detach()

        row = i // grid_cols
        col = i % grid_cols

        # Convert image to PIL and numpy array
        img_pil = F.to_pil_image(img[0], mode="L")
        img_array = np.asarray(img_pil)

        # Convert mask to PIL and numpy array
        mask_pil = F.to_pil_image(mask[0], mode="L")
        mask_array = np.asarray(mask_pil)

        # Convert to grayscale if needed
        if img_array.ndim == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)

        if mask_array.ndim == 2:
            mask_array = np.stack((mask_array,) * 3, axis=-1)

        # Add image
        fig.add_layout_image(
            dict(
                source=Image.fromarray(img_array),
                xref="x",
                yref="y",
                x=2 * col,
                y=grid_rows - row - 1, #invert y axis so images are displayed correctly.
                sizex=1,
                sizey=1,
                sizing="stretch",
                layer="above"
            )
        )

        # Add mask
        fig.add_layout_image(
            dict(
                source=Image.fromarray(mask_array),
                xref="x",
                yref="y",
                x=2 * col + 1,
                y=grid_rows - row - 1, #invert y axis so images are displayed correctly.
                sizex=1,
                sizey=1,
                sizing="stretch",
                layer="above"
            )
        )

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 2 * grid_cols]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, grid_rows]),
        margin=dict(l=0, r=0, t=0, b=0),
        width=300 * grid_cols,
        height=120 * grid_rows,
    )

    fig.show()
