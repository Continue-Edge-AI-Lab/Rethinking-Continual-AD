"""
inference.py — Measures inference time, parameter counts, and per-task storage
               for all CAD methods on edge devices.

Setup:
    This project uses uv for dependency management. To set up:

    1. Install uv (if not already installed):
        curl -LsSf https://astral.sh/uv/install.sh | sh

    2. Sync the project environment from the project root:
        cd Continual_Anomaly_Detection/
        uv sync

    3. PyTorch is NOT listed in pyproject.toml (installed separately).
       Install the version appropriate for your device:

        # CPU only (Raspberry Pi / Mac):
        uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

        # CUDA (NVIDIA Jetson — check your JetPack version for the right wheel):
        # Jetson devices use NVIDIA's PyTorch builds, not the standard PyPI ones.
        # See: https://developer.nvidia.com/embedded/downloads
        # For JetPack 6.x (L4T R36), install the matching .whl from NVIDIA, e.g.:
        #   uv pip install torch-2.x.x+nv... torchvision-0.x.x+nv...

    4. DINOv3 weights must be present at Methods/DINO/dinov3_weights/.
       EfficientAD teacher weights must be present at Methods/EfficientAD/teacher.pth.
       FastSAM weights (FastSAM-s.pt) are NOT needed — UCAD uses inference_only mode.

Usage:
    uv run python -m utils.inference

    Run from the project root (Continual_Anomaly_Detection/).
    Expects trained final weights in ../models/ for each method.

    Measures the FULL inference pipeline for each model (not just forward pass),
    including memory bank lookups, anomaly scoring, and task identification.

Output:
    - Prints a table of results to stdout
    - Saves results to results/edge_inference.csv
"""

import time
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# ============================================================
# Model imports
# ============================================================
from Methods.DNE.dne import DNE_Model
from Methods.IUF.iuf import IUF_Model
from Methods.UCAD.ucad import UCAD_Model
from Methods.PatchCore.patchcore import Patchcore_Model
from Methods.EfficientAD.efficient_ad import EfficientAD_Model
from Methods.DINO.DINOSaur import DINOSaur_Model

# ============================================================
# Config
# ============================================================
# Hardcoded weight paths (MVTEC unsupervised final weights)
# UCAD path is a "virtual" path — UCAD splits on "weights" to find its 3 memory files.
# The actual files are: ..._final_key_memory.pth, ..._final_knowledge_memory.pth, ..._final_prompt_memory.pth
WEIGHT_PATHS = {
    "DNE":         "models/DNE/DNE_MVTEC_unsupervised_final_weights.pth",
    "IUF":         "models/IUF/IUF_MVTEC_unsupervised_final_weights.pth",
    "UCAD":        "models/UCAD/UCAD_MVTEC_unsupervised_final_weights.pth",
    "PatchCore":   "models/Patchcore/Patchcore_MVTEC_unsupervised_final_weights.pth",
    "EfficientAD": "models/EfficientAD/EfficientAD_MVTEC_unsupervised_final_weights.pth",
    "DINOSaur":    "models/DINOSaur/DINOSaur_MVTEC_unsupervised_final_weights.pth",
}

N_WARMUP = 5   # Warmup runs (not timed) to stabilize hardware
N_RUNS   = 30  # Timed runs to compute mean and std

# Map model type to expected input size
INPUT_SIZE = {
    "DNE":         (3, 224, 224),
    "IUF":         (3, 224, 224),
    "UCAD":        (3, 224, 224),
    "PatchCore":   (3, 224, 224),
    "EfficientAD": (3, 256, 256),
    "DINOSaur":    (3, 224, 224),
}


# ============================================================
# Model instantiation + weight loading
# ============================================================
def check_weights_exist(model_type: str) -> bool:
    """
    Verify that the required weight files exist for a given model.
    UCAD is special — it doesn't have a single weights file, but instead
    stores 3 separate memory files (key, prompt, knowledge).
    """
    path = WEIGHT_PATHS[model_type]

    if model_type == "UCAD":
        # UCAD stores memory banks as 3 separate files
        prefix = path.split("weights")[0]
        files = [
            prefix + "key_memory.pth",
            prefix + "prompt_memory.pth",
            prefix + "knowledge_memory.pth",
        ]
        for f in files:
            if not os.path.exists(f):
                print(f"  Missing UCAD file: {f}")
                return False
        return True
    else:
        if not os.path.exists(path):
            print(f"  Weight file not found: {path}")
            return False
        # DNE also needs a companion memory file
        if model_type == "DNE":
            memory_path = path.split("weights")[0] + "memory.pth"
            if not os.path.exists(memory_path):
                print(f"  DNE memory file not found: {memory_path}")
                return False
        return True


def load_model(model_type: str) -> torch.nn.Module:
    """
    Instantiate a model and load its trained weights.
    Also performs any required post-load setup (e.g. generating
    distributions for DNE).
    """
    path = WEIGHT_PATHS[model_type]

    match model_type:
        case "DNE":
            model = DNE_Model()
            model.load(path)
            # DNE needs global distribution built from memory before predict()
            model.generate_global_dist()

        case "IUF":
            # IUF needs num_tasks to match training config (15 for MVTEC)
            model = IUF_Model(num_tasks=15)
            model.load(path)

        case "UCAD":
            # inference_only=True skips FastSAM loading (not needed for inference)
            model = UCAD_Model(vit_output_layer=5, inference_only=True)
            model.load(path)

        case "PatchCore":
            model = Patchcore_Model()
            model.load(path)

        case "EfficientAD":
            model = EfficientAD_Model(edge_testing=True)
            model.load(path)

        case "DINOSaur":
            model = DINOSaur_Model()
            model.load(path)

    model.eval()
    return model


# ============================================================
# Single-image inference wrappers
#
# Each function takes a model and a single input image tensor
# (C, H, W) and returns an image-level anomaly score (float).
# These replicate the FULL scoring pipeline from each model's
# eval code — not just the forward pass.
# ============================================================

def infer_dne(model: DNE_Model, img: torch.Tensor) -> float:
    """
    DNE: forward → embedding → Mahalanobis distance against global distribution.
    Uses DNE_Model.predict() which handles batching internally via embed().
    """
    with torch.no_grad():
        score = model.predict(img.unsqueeze(0))
    return score


def infer_iuf(model: IUF_Model, img: torch.Tensor) -> float:
    """
    IUF: forward → reconstruction → L2 reconstruction error → sum.
    Mirrors the scoring logic in IUF_Model.calc_results().
    """
    with torch.no_grad():
        x = img.unsqueeze(0).to(model.device)
        x_recon, _, _ = model.forward(x)
        recon_error = (x - x_recon).abs()
        score = recon_error.sum().item()
    return score


def infer_efficientad(model: EfficientAD_Model, img: torch.Tensor) -> float:
    """
    EfficientAD: forward → teacher/student/autoencoder features → MSE anomaly maps → max.
    Mirrors the scoring logic in EfficientAD_Model.eval_one_epoch().
    """
    with torch.no_grad():
        x = img.unsqueeze(0).to(model.device)
        t_features, s_features, ae_features = model.forward(x)

        # Combined anomaly map (same as eval_one_epoch)
        m_st = F.mse_loss(t_features, s_features[:, 0:384, :, :], reduction='none').mean(dim=1)
        m_stae = F.mse_loss(ae_features, s_features[:, 384:768, :, :], reduction='none').mean(dim=1)
        anomaly_map = 0.5 * m_stae + 0.5 * m_st  # (1, 64, 64)
        anomaly_map = model.anomaly_map_upsample(anomaly_map)  # (1, 256, 256)
        score = anomaly_map.flatten().max().item()
    return score


def infer_patchcore(model: Patchcore_Model, img: torch.Tensor) -> float:
    """
    PatchCore: forward → patch features → KNN against memory bank with re-weighting.
    Uses Patchcore_Model.predict(). Returns image-level score.
    """
    with torch.no_grad():
        scores = model.predict(img.unsqueeze(0))
    # predict() returns a list of image-level scores
    return scores[0]


def infer_ucad(model: UCAD_Model, img: torch.Tensor) -> float:
    """
    UCAD: forward (no prompt) → task ID via key memory → forward (with prompt)
          → KNN against knowledge memory → re-weighted patch scoring → max.
    Mirrors the scoring logic in UCAD_Model.eval_one_epoch().
    """
    with torch.no_grad():
        x = img.unsqueeze(0).to(model.device)

        # Step 1: Task identification (forward without prompt)
        img_features = model.forward(x, with_prompt=False)  # (1, 196, 768)
        embed = img_features[0]  # (196, 768)

        # Find closest task key (Equation 4)
        sim_scores = []
        for k in range(len(model.key_memory)):
            test_key = model.key_memory[k]
            sim = 0
            for patch in range(196):
                patch_dists = F.pairwise_distance(
                    embed[patch].unsqueeze(0),
                    test_key.to(model.device)
                )
                sim += patch_dists.min()
            sim_scores.append(sim)
        chosen_task = torch.stack(sim_scores).argmin().item()

        # Step 2: Forward with chosen task prompt
        model.prompt = nn.Parameter(model.prompt_memory[chosen_task].clone().detach())
        img_knowledge = model.forward(x, with_prompt=True).squeeze(0)  # (196, 768)
        task_knowledge = model.knowledge_memory[chosen_task].to(model.device)  # (196, 768)

        # Step 3: Anomaly scoring (Equations 5 and 6 from the paper)
        m_star_idx = []
        m_star_dist = []
        for patch in range(196):
            test_patch = img_knowledge[patch]
            dists = F.pairwise_distance(task_knowledge, test_patch)
            min_idx = dists.argmin().item()
            m_star_idx.append(min_idx)
            m_star_dist.append(dists.min().item())

        m_star_idx = torch.tensor(m_star_idx).to(model.device)
        base_scores = torch.tensor(m_star_dist).to(model.device)

        # Re-weighting (Equation 6)
        num_neighbors = 5
        final_s_scores = []
        for patch in range(196):
            test_patch = img_knowledge[patch]
            nearest_knowledge = task_knowledge[m_star_idx[patch]]
            nn_dists = F.pairwise_distance(task_knowledge, nearest_knowledge)
            _, neighbor_indices = torch.topk(nn_dists, k=num_neighbors, largest=False)
            neighbors = task_knowledge[neighbor_indices]
            dists_to_neighbors = F.pairwise_distance(neighbors, test_patch)

            numerator = torch.exp(base_scores[patch])
            denominator = torch.clamp(torch.exp(dists_to_neighbors).sum(), min=1e-5)
            weight = 1.0 - (numerator / denominator)
            final_s_scores.append(weight * base_scores[patch])

        final_s_scores = torch.stack(final_s_scores)
        score = final_s_scores.max().item()

    return score


def infer_dinosaur(model: DINOSaur_Model, img: torch.Tensor) -> float:
    """
    DINOSaur: forward → CLS token task ID → neighborhood-restricted KNN → max patch score.
    Uses DINOSaur_Model.predict(). Returns image-level score (max of patch scores).
    """
    with torch.no_grad():
        patch_scores = model.predict(img.unsqueeze(0), neighborhood=3)
    # predict() returns (1, 14, 14) tensor of patch-level scores
    return patch_scores.max().item()


# Map model type to its inference function
INFER_FN = {
    "DNE":         infer_dne,
    "IUF":         infer_iuf,
    "UCAD":        infer_ucad,
    "PatchCore":   infer_patchcore,
    "EfficientAD": infer_efficientad,
    "DINOSaur":    infer_dinosaur,
}


# ============================================================
# Parameter counting
# ============================================================
def count_params(model: torch.nn.Module) -> dict:
    """
    Count total and trainable parameters.
    Returns dict with 'total' and 'trainable' counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _tensor_bytes(t: torch.Tensor) -> int:
    """Return the storage size in bytes of a tensor."""
    return t.nelement() * t.element_size()


def measure_per_task_storage(model, model_type: str) -> dict:
    """
    Measure per-task memory storage for each model.

    For memory-based methods (DINOSaur, PatchCore, UCAD), this computes
    the average bytes stored per task in their memory banks. These grow
    linearly as new tasks are added — a critical metric for edge deployment.

    For non-memory methods (DNE, IUF, EfficientAD), per-task storage is
    zero since model weights update in-place without growing.

    Returns dict with:
        'total_memory_bytes': total size of all task-specific memory
        'num_tasks': number of tasks stored
        'per_task_bytes': average bytes per task
    """
    match model_type:
        case "DINOSaur":
            # cls_memory: dict {task_name: tensor (384,)} — one CLS prototype per task
            # patch_memory: dict {task_name: tensor (N, 14, 14, 384)} — coreset per task
            total = 0
            num_tasks = len(model.patch_memory)
            for task in model.patch_memory:
                total += _tensor_bytes(model.cls_memory[task])
                total += _tensor_bytes(model.patch_memory[task])
            return {
                "total_memory_bytes": total,
                "num_tasks": num_tasks,
                "per_task_bytes": total / max(num_tasks, 1),
            }

        case "PatchCore":
            # memory_bank: single tensor that grows as coresets are appended per task
            # We know the number of tasks from the training config (15 for MVTEC)
            num_tasks = 15  # MVTEC unsupervised
            total = _tensor_bytes(model.memory_bank) if model.memory_bank is not None else 0
            return {
                "total_memory_bytes": total,
                "num_tasks": num_tasks,
                "per_task_bytes": total / max(num_tasks, 1),
            }

        case "UCAD":
            # key_memory, prompt_memory, knowledge_memory: lists, one entry per task
            total = 0
            num_tasks = len(model.key_memory)
            for i in range(num_tasks):
                total += _tensor_bytes(model.key_memory[i])
                total += _tensor_bytes(model.prompt_memory[i])
                total += _tensor_bytes(model.knowledge_memory[i])
            return {
                "total_memory_bytes": total,
                "num_tasks": num_tasks,
                "per_task_bytes": total / max(num_tasks, 1),
            }

        case "DNE" | "IUF" | "EfficientAD":
            # Non-memory methods: weights update in-place, no per-task growth
            return {
                "total_memory_bytes": 0,
                "num_tasks": 0,
                "per_task_bytes": 0,
            }


# ============================================================
# Validation — run a single dummy input to verify the pipeline
# ============================================================
def validate_prediction(model, model_type: str) -> float:
    """
    Run one dummy image through the full inference pipeline.
    Returns the anomaly score. Raises an exception if anything fails.
    """
    device = next(model.parameters()).device
    img_size = INPUT_SIZE[model_type]
    dummy_img = torch.randn(img_size, device=device)
    infer_fn = INFER_FN[model_type]
    score = infer_fn(model, dummy_img)
    # Ensure we return a plain Python float (some models return a Tensor)
    if isinstance(score, torch.Tensor):
        score = score.item()
    return float(score)


# ============================================================
# Timing
# ============================================================
def time_inference(model, model_type: str, n_warmup: int = N_WARMUP, n_runs: int = N_RUNS) -> dict:
    """
    Time the full inference pipeline for a single image.

    Performs n_warmup untimed runs, then n_runs timed runs.
    Uses torch.cuda.synchronize() for accurate GPU timing.

    Returns dict with 'mean_ms', 'std_ms', and 'all_times_ms'.
    """
    device = next(model.parameters()).device
    use_cuda = (device.type == "cuda")

    # Create a random input image of the correct size for this model
    img_size = INPUT_SIZE[model_type]
    img = torch.randn(img_size, device=device)

    infer_fn = INFER_FN[model_type]

    # Warmup
    for _ in range(n_warmup):
        _ = infer_fn(model, img)
        if use_cuda:
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()

        _ = infer_fn(model, img)

        if use_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "all_times_ms": times,
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Edge Inference Profiling — Continual Anomaly Detection Methods")
    print("=" * 70)

    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "Apple MPS"
    print(f"Device: {device_name}")
    print(f"Warmup runs: {N_WARMUP} | Timed runs: {N_RUNS}")
    print()

    results = []
    model_order = [
        "DNE",
       "IUF",
       "UCAD",
       "PatchCore",
       "EfficientAD",
       "DINOSaur"
    ]

    for model_type in model_order:
        print(f"--- {model_type} ---")

        # Check that weight files exist before trying to load
        if not check_weights_exist(model_type):
            print(f"  SKIPPED (missing weight files)\n")
            results.append({
                "method": model_type,
                "total_params_M": None,
                "trainable_params_M": None,
                "per_task_storage_MB": None,
                "total_task_memory_MB": None,
                "num_tasks": None,
                "inference_mean_ms": None,
                "inference_std_ms": None,
                "validation_score": None,
            })
            continue

        # Load model
        print(f"  Loading weights...")
        try:
            model = load_model(model_type)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            results.append({
                "method": model_type,
                "total_params_M": None,
                "trainable_params_M": None,
                "per_task_storage_MB": None,
                "total_task_memory_MB": None,
                "num_tasks": None,
                "inference_mean_ms": None,
                "inference_std_ms": None,
                "validation_score": None,
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # Count params
        params = count_params(model)
        total_M = params["total"] / 1e6
        trainable_M = params["trainable"] / 1e6
        print(f"  Parameters: {total_M:.2f}M total, {trainable_M:.2f}M trainable")

        # Measure per-task storage
        storage = measure_per_task_storage(model, model_type)
        per_task_MB = storage["per_task_bytes"] / (1024 * 1024)
        total_mem_MB = storage["total_memory_bytes"] / (1024 * 1024)
        if storage["num_tasks"] > 0:
            print(f"  Task memory: {total_mem_MB:.2f} MB total across {storage['num_tasks']} tasks "
                  f"({per_task_MB:.2f} MB/task)")
        else:
            print(f"  Task memory: N/A (weights update in-place, no per-task growth)")

        # Validate with a single dummy input
        print(f"  Validating with dummy input...")
        try:
            val_score = validate_prediction(model, model_type)
            print(f"  Validation OK — dummy score: {val_score:.4f}")
        except Exception as e:
            print(f"  VALIDATION FAILED: {e}")
            results.append({
                "method": model_type,
                "total_params_M": round(total_M, 2),
                "trainable_params_M": round(trainable_M, 2),
                "per_task_storage_MB": round(per_task_MB, 2),
                "total_task_memory_MB": round(total_mem_MB, 2),
                "num_tasks": storage["num_tasks"],
                "inference_mean_ms": None,
                "inference_std_ms": None,
                "validation_score": None,
            })
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # Time inference
        print(f"  Timing inference ({N_WARMUP} warmup + {N_RUNS} timed runs)...")
        timing = time_inference(model, model_type)
        print(f"  Inference: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms/image")

        results.append({
            "method": model_type,
            "total_params_M": round(total_M, 2),
            "trainable_params_M": round(trainable_M, 2),
            "per_task_storage_MB": round(per_task_MB, 2),
            "total_task_memory_MB": round(total_mem_MB, 2),
            "num_tasks": storage["num_tasks"],
            "inference_mean_ms": round(timing["mean_ms"], 2),
            "inference_std_ms": round(timing["std_ms"], 2),
            "validation_score": round(val_score, 6),
        })

        # Free memory before next model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print()

    # Print summary table
    print("=" * 100)
    print(f"{'Method':<15} {'Params (M)':>12} {'Trainable (M)':>15} {'MB/Task':>10} {'Inference (ms)':>18} {'Valid.':>10}")
    print("-" * 100)
    for r in results:
        if r["inference_mean_ms"] is not None:
            inf_str = f"{r['inference_mean_ms']:.2f} ± {r['inference_std_ms']:.2f}"
            task_str = f"{r['per_task_storage_MB']:.2f}" if r['per_task_storage_MB'] > 0 else "N/A"
            val_str = f"{r['validation_score']:.4f}"
            print(f"{r['method']:<15} {r['total_params_M']:>12.2f} {r['trainable_params_M']:>15.2f} {task_str:>10} {inf_str:>18} {val_str:>10}")
        else:
            status = "SKIP" if r["total_params_M"] is None else "FAIL"
            print(f"{r['method']:<15} {status:>12}")
    print("=" * 100)

    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = "./results/edge_inference.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
