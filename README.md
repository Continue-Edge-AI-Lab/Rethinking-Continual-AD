# Rethinking Continual Anomaly Detection on the Edge

Official code for our ECCV 2026 paper, **"Rethinking Continual Anomaly Detection on the Edge: Benchmarking Under Realistic Industrial Conditions."**

This repository contains a unified benchmark for Continual Anomaly Detection (CAD) and our method, **DINOSaur**, a training-free detector built on a frozen DINOv3 backbone with spatially-indexed coreset memory and neighborhood-restricted scoring. The benchmark includes a discrete-task protocol, a continuous-drift protocol, evaluation on logical anomalies, and efficiency profiling on edge hardware.

Paper: [arXiv preprint](https://arxiv.org/abs/2605.24251).

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment management and is pinned to Python 3.13.

1. **Install uv** (see the [uv install guide](https://docs.astral.sh/uv/getting-started/installation/)).

2. **Create the environment and install dependencies:**

   ```bash
   uv sync
   ```

   This creates a `.venv` (Python 3.13) and installs the listed dependencies (einops, ultralytics, numpy, pandas, scikit-learn, etc.).

3. **Install PyTorch separately, matched to your hardware.** PyTorch is intentionally not pinned in `pyproject.toml`, because the correct build depends on your GPU. Follow the official selector at [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and install the build for your platform:

   ```bash
   # Example only — use the exact command from the PyTorch site for your setup.
   # NVIDIA (CUDA):
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   # AMD (ROCm):
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
   # CPU only:
   uv pip install torch torchvision
   ```

   Installing the hardware-specific build here overrides any default PyTorch that may have been pulled in as a transitive dependency.

4. **Activate the environment:**

   ```bash
   source .venv/bin/activate
   ```

5. **Run:**

   ```bash
   python main.py
   ```

---

## DINOv3 backbone (required for DINOSaur)

DINOSaur uses a frozen **DINOv3 ViT-S/16** backbone. We do not redistribute the DINOv3 source or weights, since they are released under Meta's own [DINOv3 License](https://github.com/facebookresearch/dinov3). You provide them locally in two steps.

1. **Clone the DINOv3 source** into `Methods/DINO/dinov3`:

   ```bash
   git clone https://github.com/facebookresearch/dinov3 Methods/DINO/dinov3
   ```

2. **Download the ViT-S/16 weights** (gated; you must accept Meta's license) and place the file in `Methods/DINO/dinov3_weights/` with its original name:

   ```bash
   mkdir -p Methods/DINO/dinov3_weights
   # Place the downloaded checkpoint here, named exactly:
   #   Methods/DINO/dinov3_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
   ```

   The weights are available from the [DINOv3 repository](https://github.com/facebookresearch/dinov3) and on Hugging Face (`facebook/dinov3-vits16-pretrain-lvd1689m`). The filename must match the entry in `Methods/DINO/get_dino_model.py`.

   Our code loads the backbone with `torch.hub.load(..., source='local')`, so the cloned source and the checkpoint above are all that is needed.

### Using timm instead (optional)

Since our paper's release, DINOv3 backbones have also been made available through [timm](https://github.com/huggingface/pytorch-image-models) (for example, `vit_small_patch16_dinov3.lvd1689m`), which avoids the manual clone and download. We did not use this route in the paper, but you may prefer it. If you do, you must match the exact interface DINOSaur expects and adapt `DINOSaur.forward()` to timm's output format.

DINOSaur requires, for a `(B, 3, 224, 224)` input batch:

- **Patch size 16**, giving a **14 x 14 = 196** patch-token grid.
- **Embedding dimension 384** (ViT-S/16).
- A **CLS token** of dimension **384**.

So `forward()` must return a CLS token of shape `(B, 384)` and patch tokens of shape `(B, 196, 384)`. Note that timm does not expose the `x_norm_clstoken` / `x_norm_patchtokens` dictionary that the Meta `forward_features` API returns, so you will need to extract and L2-normalize the CLS and patch tokens yourself before passing them on. Any backbone you swap in must produce these same dimensions for the rest of the pipeline (coreset memory, neighborhood scoring) to work unchanged.

---

## FastSAM (used by UCAD)

The UCAD baseline uses FastSAM during training. The weights file (`FastSAM-s.pt`) is **downloaded automatically by ultralytics on first use**, so no manual step is required, but the first UCAD run needs an internet connection.

---

## Datasets

We provide our datasets, including both **supervised** and **unsupervised** scenarios, here:

**[Google Drive: datasets](https://drive.google.com/drive/folders/1oGlq_BEtlMTZ1WU2JuQ584X7Ak1hzWu6)**

To use them, download the folders from the link and place them where the dataset loaders in `datasets/` expect them (see the path settings at the top of `datasets/mvtec.py`, `datasets/mvtec_loco.py`, and `datasets/mtd.py`). The Drive link is set to "anyone with the link," so no special access request is needed.

**Attribution and licensing.** Our benchmark builds on existing datasets, each under its own license. Please cite and comply with the original terms:

- MVTec-AD and MVTec-LOCO are released under CC BY-NC-SA 4.0 (non-commercial). See [MVTec](https://www.mvtec.com/company/research/datasets).
- Our continuous-drift data is derived from the Magnetic Tile Defects (MTD) dataset; see its [source](https://github.com/abin24/Magnetic-tile-defect-datasets).

The non-commercial terms of MVTec apply to any derivatives. Please use the provided data for research purposes only.

---

## Usage

All experiments are configured at the top of `main.py`:

- `TRAIN` / `EVAL`: toggle training and evaluation.
- `models`: the set of methods to run (DNE, IUF, UCAD, Patchcore, EfficientAD, DINOSaur).
- `datasets`: which benchmarks to run (MVTEC, MVTEC_LOCO, MTD).
- `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`: training hyperparameters.
- `EVAL_METRICS`: which metrics to compute (for example, `pixel_auroc`).

By default the repo trains the methods from scratch, so no pretrained method weights are required (the DINOv3 backbone and FastSAM above are still needed). Trained weights are written under `models/`, and evaluation results under `results/`.

---

## Citation

If you use this benchmark or DINOSaur, please cite:

```bibtex
@inproceedings{weatherly2026rethinking,
  title     = {Rethinking Continual Anomaly Detection on the Edge:
               Benchmarking Under Realistic Industrial Conditions},
  author    = {Weatherly, Chad and Lin, Sen},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

The proceedings volume and page numbers will be added once the ECCV 2026 proceedings are published.

---

## License

The **code** in this repository is released under the Apache License 2.0 (see `LICENSE`). The **datasets** are subject to the licenses of their original sources, as described in the Datasets section above. The DINOv3 backbone is subject to Meta's DINOv3 License.
