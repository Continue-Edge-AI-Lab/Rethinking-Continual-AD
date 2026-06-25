"""
Figure generation for the ECCV paper.

Usage:
    python utils/figures.py              # generates all figures
    python utils/figures.py --mtd        # generates only the MTD augmentation figure

All outputs are saved to the figures/ directory as both PDF (for LaTeX) and PNG (for quick viewing).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import argparse
import os
from torchvision.io import read_image
import torchvision.transforms.v2.functional as F
from matplotlib.patches import FancyArrowPatch


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────
# MTD Progressive Augmentation Figure
# ─────────────────────────────────────────────────────────────

def apply_color(img, params):
    """Apply color distortion (brightness, contrast, saturation).
    Uses the midpoint of the parameter window for a deterministic sample."""
    value = (params[0] + params[1]) / 2
    factor = 1 + value
    out = F.adjust_brightness(img, factor)
    out = F.adjust_contrast(out, factor)
    out = F.adjust_saturation(out, factor)
    return out


def apply_blur(img, params):
    """Apply Gaussian blur with given kernel size and sigma."""
    kernel_size = int(params[0])
    if kernel_size < 1:
        return img
    if kernel_size % 2 == 0:  # kernel must be odd
        kernel_size += 1
    sigma = float(params[1])
    return F.gaussian_blur(img, kernel_size, [sigma, sigma])


def apply_geometric(img, params):
    """Apply geometric distortion (rotation, translation, scale, shear).
    Uses fixed (non-random) values so the figure is deterministic."""
    degrees, translate, scale, shear = params
    return F.affine(
        img,
        angle=float(degrees),
        translate=[float(translate), float(translate)],
        scale=1.0 + float(scale),
        shear=float(shear)
    )


def to_numpy(tensor):
    """Convert a CHW uint8 tensor to an HWC float numpy array in [0, 1]."""
    arr = tensor.permute(1, 2, 0).numpy()
    return np.clip(arr / 255.0 if arr.max() > 1 else arr, 0, 1)


def generate_mtd_augmentation_figure(output_dir='figures'):
    """
    Generate a 3x6 grid figure showing progressive augmentation on MTD tiles.
    Rows: Color Distortion, Gaussian Blur, Geometric Distortion
    Columns: Original, Task 1, Task 3, Task 5, Task 7, Task 10
    """
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)

    # ── Load source image ──
    src_path = 'datasets/magnetic_tile_defects/unsupervised/train/MT_Free/exp1_num_110086.jpg'
    img_raw = read_image(src_path).expand(3, -1, -1)  # ensure 3 channels

    # ── Augmentation parameters (must match main.py) ──
    color_params = [
        [0.0, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2], [0.2, 0.25],
        [0.25, 0.3], [0.3, 0.35], [0.35, 0.4], [0.4, 0.45], [0.45, 0.5]
    ]
    blur_params = [
        [1, 0.5], [3, 1], [5, 1.5], [7, 2], [9, 2.5],
        [11, 3], [13, 3.5], [15, 4], [17, 4.5], [19, 5]
    ]
    geometric_params = [
        [2, 1, 0.01, 1], [4, 2, 0.02, 2], [6, 3, 0.03, 3], [8, 4, 0.04, 4],
        [10, 5, 0.05, 5], [12, 6, 0.06, 6], [14, 7, 0.07, 7], [16, 8, 0.08, 8],
        [18, 9, 0.09, 9], [20, 10, 0.10, 10]
    ]

    # Show Original + Tasks 1, 3, 5, 7, 10 (0-indexed into params)
    task_indices = [0, 2, 4, 6, 9]
    task_labels = ['Task 1', 'Task 3', 'Task 5', 'Task 7', 'Task 10']

    # ── Generate augmented images ──
    color_imgs = [to_numpy(apply_color(img_raw, color_params[i])) for i in task_indices]
    blur_imgs = [to_numpy(apply_blur(img_raw, blur_params[i])) for i in task_indices]
    geo_imgs = [to_numpy(apply_geometric(img_raw, geometric_params[i])) for i in task_indices]
    orig_img = to_numpy(img_raw)

    # ── Assemble grid data ──
    n_img_cols = 6  # Original + 5 tasks
    section_names = ['Color Distortion', 'Gaussian Blur', 'Geometric Distortion']
    col_headers = ['Original'] + task_labels
    all_imgs = [
        [orig_img] + color_imgs,
        [orig_img] + blur_imgs,
        [orig_img] + geo_imgs,
    ]
    all_params = [
        [''] + [f'w=[{color_params[i][0]:.2f}, {color_params[i][1]:.2f}]' for i in task_indices],
        [''] + [f'k={int(blur_params[i][0])}, \u03c3={blur_params[i][1]:.1f}' for i in task_indices],
        [''] + [f'\u03b8={geometric_params[i][0]}\u00b0, s={geometric_params[i][2]:.2f}' for i in task_indices],
    ]

    # ── Font setup to match ECCV/LNCS paper (Computer Modern serif) ──
    FONT = 'serif'  # matches LaTeX Computer Modern in the paper

    # ── Create figure ──
    # Compact layout: reduced height, tighter spacing, larger fonts
    fig, axes = plt.subplots(3, n_img_cols, figsize=(12, 7.0), dpi=300, facecolor='white')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.01,
                        wspace=0.05, hspace=0.55)

    for r in range(3):
        for c in range(n_img_cols):
            ax = axes[r, c]
            ax.imshow(all_imgs[r][c], cmap='gray', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
                sp.set_color('#bbbbbb')

            # Column headers (first row only) — placed close above the images
            if r == 0:
                ax.set_title(col_headers[c], fontsize=14, fontweight='bold', pad=6,
                             fontfamily=FONT)

            # Parameter annotations below each augmented image
            if c > 0:
                ax.text(0.5, -0.10, all_params[r][c],
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=11, color='#555555', fontfamily=FONT)

        # Section label above each row, centered across the full row
        left_pos = axes[r, 0].get_position().x0
        right_pos = axes[r, -1].get_position().x1
        mid_x = (left_pos + right_pos) / 2
        top_y = axes[r, 0].get_position().y1
        # For row 0, place section label higher to clear the column headers
        offset = 0.045 if r == 0 else 0.025
        fig.text(mid_x, top_y + offset, section_names[r],
                 ha='center', va='bottom',
                 fontsize=14, fontweight='bold', fontfamily=FONT)

    # ── Arrow showing drift direction ──
    arrow = FancyArrowPatch(
        (0.12, 0.94), (0.96, 0.94),
        transform=fig.transFigure, arrowstyle='->', color='#333333',
        lw=1.5, mutation_scale=16, clip_on=False
    )
    fig.patches.append(arrow)
    fig.text(0.54, 0.955, 'Increasing Drift Intensity',
             ha='center', va='bottom', fontsize=14, fontweight='bold',
             color='#333333', fontfamily=FONT)

    # ── Save ──
    pdf_path = os.path.join(output_dir, 'mtd_augmentation.pdf')
    png_path = os.path.join(output_dir, 'mtd_augmentation.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'MTD augmentation figure saved to {pdf_path} and {png_path}')


# ─────────────────────────────────────────────────────────────
# Main — run from project root: python utils/figures.py
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate paper figures.')
    parser.add_argument('--mtd', action='store_true', help='Generate MTD augmentation figure')
    parser.add_argument('--all', action='store_true', help='Generate all figures')
    parser.add_argument('--output-dir', type=str, default='figures', help='Output directory')
    args = parser.parse_args()

    # Default to --all if no specific figure is requested
    if not args.mtd:
        args.all = True

    if args.all or args.mtd:
        generate_mtd_augmentation_figure(args.output_dir)

    print('Done.')
