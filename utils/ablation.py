"""
ablation.py — DINOSaur ablation studies for coreset percentage and neighborhood size.

Usage:
    # Train only (run on PC from project root):
    python -m utils.ablation -t
    python -m utils.ablation -t --datasets MVTEC MVTEC_LOCO
    python -m utils.ablation -t --coreset_pcts 0.1 0.2 0.3

    # Evaluate only (after training is done):
    python -m utils.ablation -e
    python -m utils.ablation -e --datasets MVTEC --coreset_pcts 0.05 0.1

    # Aggregate per-task results into summary tables:
    python -m utils.ablation -a
    python -m utils.ablation -a --datasets MVTEC

    # Inference profiling (run on edge devices from project root):
    #   1. Copy this file to the project root
    #   2. Run:  python ablation.py -i
    python ablation.py -i
    python ablation.py -i --weights_dataset MVTEC

Design:
    - coreset_pct affects training (memory bank size), so we train once per coreset_pct per dataset
    - neighborhood affects only predict(), so we evaluate each trained model at all neighborhood values
    - Training (-t) and evaluation (-e) are separate so long runs can be resumed independently
    - With -i flag, loads trained ablation weights and profiles inference timing on edge devices

    Weight naming convention:
        models/DINOSaur/ablation/DINOSaur_{dataset}_c{pct}_weights.pth
        e.g. models/DINOSaur/ablation/DINOSaur_MVTEC_c5_weights.pth  (for coreset_pct=0.05)

Output:
    Per-task results (one folder per coreset):
        results/ablation/{dataset}/c{pct}/img_auroc.csv
        results/ablation/{dataset}/c{pct}/img_acc.csv
        results/ablation/{dataset}/c{pct}/img_recall.csv
        Each CSV has rows like "c5_n0", "c5_n1", ... and columns are task names.

    Per-dataset summaries (rows = coreset, columns = neighborhood, cells = mean across tasks):
        results/ablation/{dataset}/{dataset}_img_auroc.csv
        results/ablation/{dataset}/{dataset}_img_acc.csv
        results/ablation/{dataset}/{dataset}_img_recall.csv

    Global summaries (averaged across all datasets):
        results/ablation/img_auroc.csv
        results/ablation/img_acc.csv
        results/ablation/img_recall.csv

    Edge inference profiling:
        results/ablation/edge_inference.csv  (with -i flag)
"""

import argparse
import os
import gc
import time
import torch
import numpy as np
import pandas as pd

from Methods.DINO.DINOSaur import DINOSaur_Model

# ============================================================
# Ablation configuration
# ============================================================
CORESET_PCTS = [0.01, 0.025, 0.05, 0.1, 0.2]
NEIGHBORHOODS = [0, 1, 2, 3, 4]
METRICS = ["img_auroc", "img_acc", "img_recall"]

# Dataset configs: name → task list
DATASET_CONFIGS = {
    "MVTEC": {
        "tasks": ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                  'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                  'transistor', 'wood', 'zipper'],
    },
    "MVTEC_LOCO": {
        "tasks": ['breakfast_box', 'screw_bag', 'pushpins',
                  'splicing_connectors', 'juice_bottle'],
    },
    "MTD_color": {
        "data_aug": "color",
        "tasks": [[0.0, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2], [0.2, 0.25],
                  [0.25, 0.3], [0.3, 0.35], [0.35, 0.4], [0.4, 0.45], [0.45, 0.5]],
        "task_names": ['color_00_005', 'color_005_01', 'color_01_015', 'color_015_02',
                       'color_02_025', 'color_025_03', 'color_03_035', 'color_035_04',
                       'color_04_045', 'color_045_05'],
    },
    "MTD_blur": {
        "data_aug": "blur",
        "tasks": [[1, 0.5], [3, 1], [5, 1.5], [7, 2], [9, 2.5],
                  [11, 3], [13, 3.5], [15, 4], [17, 4.5], [19, 5]],
        "task_names": ['blur_1_05', 'blur_3_1', 'blur_5_15', 'blur_7_2', 'blur_9_25',
                       'blur_11_3', 'blur_13_35', 'blur_15_4', 'blur_17_45', 'blur_19_5'],
    },
    "MTD_geometric": {
        "data_aug": "geometric",
        "tasks": [[2, 1, 0.01, 1], [4, 2, 0.02, 2], [6, 3, 0.03, 3], [8, 4, 0.04, 4], [10, 5, 0.05, 5],
                  [12, 6, 0.06, 6], [14, 7, 0.07, 7], [16, 8, 0.08, 8], [18, 9, 0.09, 9], [20, 10, 0.10, 10]],
        # NOTE: task_names use auto-generated format from train.py:
        #   str(param).replace(".","") joined by "_"
        #   e.g. [2, 1, 0.01, 1] → "geometric_2_1_001_1"
        # The original eval CSVs used 3-param tasks and have names like "geometric_2_1001_1".
        # These new 4-param names will NOT match those old CSVs — that's expected since
        # the geometric params were corrected.
        "task_names": ['geometric_2_1_001_1', 'geometric_4_2_002_2', 'geometric_6_3_003_3',
                       'geometric_8_4_004_4', 'geometric_10_5_005_5', 'geometric_12_6_006_6',
                       'geometric_14_7_007_7', 'geometric_16_8_008_8', 'geometric_18_9_009_9',
                       'geometric_20_10_010_10'],
    },
}

# Paths — relative to project root (works from both utils/ via -m and root via -i)
ABLATION_RESULTS_DIR = "results/ablation"
ABLATION_WEIGHTS_DIR = "models/DINOSaur/ablation"


# ============================================================
# Helpers
# ============================================================
def get_task_names(dataset_name: str, config: dict) -> list:
    """Return the list of task name strings used as CSV column headers."""
    if "task_names" in config:
        return config["task_names"]
    return config["tasks"]


def fmt_coreset(coreset_pct: float) -> str:
    """Format coreset_pct for filenames. e.g. 0.05 → '05', 0.1 → '1', 0.5 → '5'."""
    # Multiply by 100 to get the percentage, format as int
    pct_int = str(coreset_pct).split('.')[1]
    return str(pct_int)


def weight_path(dataset_name: str, coreset_pct: float) -> str:
    """Return the weight file path for a given ablation config."""
    return os.path.join(ABLATION_WEIGHTS_DIR,
                        f"DINOSaur_{dataset_name}_c{fmt_coreset(coreset_pct)}_weights.pth")


def cleanup_gpu():
    """Force garbage collection and clear CUDA/ROCm cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ============================================================
# Training: build memory banks for one coreset_pct on one dataset
# ============================================================
def train_ablation(dataset_name: str, coreset_pct: float):
    """
    Train DINOSaur with given coreset_pct on all tasks of a dataset.
    Saves the final model weights.
    """
    # Lazy import — not needed on edge devices
    from torch.utils.data import DataLoader
    import datasets

    config = DATASET_CONFIGS[dataset_name]
    tasks = config["tasks"]
    is_mtd = dataset_name.startswith("MTD")

    print(f"\n{'='*60}")
    print(f"Training: {dataset_name} | coreset_pct={coreset_pct}")
    print(f"{'='*60}")

    model = DINOSaur_Model(coreset_pct=coreset_pct)

    for t, task in enumerate(tasks):
        if is_mtd:
            task_name = config["task_names"][t]
        else:
            task_name = task

        print(f"  Task {t+1}/{len(tasks)}: {task_name}")

        # Build dataloader
        if dataset_name == "MVTEC":
            task_dataset = datasets.mvtec(train=True, unsupervised=True,
                                          replay=0, task=task,
                                          transform=None)
        elif dataset_name == "MVTEC_LOCO":
            task_dataset = datasets.mvtec_loco(train=True, task=task,
                                               replay=0, transform=None)
        elif is_mtd:
            task_dataset = datasets.mtd(train=True, unsupervised=True,
                                        replay=True, transform=None,
                                        data_aug=config["data_aug"],
                                        data_aug_params=task)

        dataloader = DataLoader(task_dataset, batch_size=32,
                                shuffle=True, collate_fn=datasets.collate)

        # Train one epoch (DINOSaur only needs 1 — frozen backbone, just builds memory)
        model.train_one_epoch(dataset=task_dataset,
                              dataloader=dataloader,
                              task_name=task_name,
                              optimizer=None,
                              criterion=None,
                              task_num=(t+1))

        # Free dataset/dataloader memory between tasks
        del task_dataset, dataloader
        cleanup_gpu()

    # Save model with unique name
    save_path = weight_path(dataset_name, coreset_pct)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"  Saved: {save_path}")

    del model
    cleanup_gpu()


# ============================================================
# Evaluation: evaluate one trained model at all neighborhoods
# ============================================================
def evaluate_ablation(dataset_name: str, coreset_pct: float):
    """
    Load a trained ablation model and evaluate at all neighborhood values.
    Saves per-metric CSVs to results/ablation/{dataset}/.
    """
    # Lazy import — not needed on edge devices
    from torch.utils.data import DataLoader
    import datasets
    import torcheval.metrics.functional as tef

    config = DATASET_CONFIGS[dataset_name]
    tasks = config["tasks"]
    task_names = get_task_names(dataset_name, config)
    is_mtd = dataset_name.startswith("MTD")

    # Output directory — one folder per coreset percentage
    coreset_tag = f"c{fmt_coreset(coreset_pct)}"
    out_dir = os.path.join(ABLATION_RESULTS_DIR, dataset_name, coreset_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    wp = weight_path(dataset_name, coreset_pct)
    if not os.path.exists(wp):
        print(f"  SKIP eval: weights not found at {wp}")
        return

    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name} | coreset_pct={coreset_pct}")
    print(f"{'='*60}")

    model = DINOSaur_Model(coreset_pct=coreset_pct)
    model.load(wp)
    model.eval()

    for neighborhood in NEIGHBORHOODS:
        print(f"\n  Neighborhood={neighborhood}")

        row_name = f"c{fmt_coreset(coreset_pct)}_n{neighborhood}"

        for t, task in enumerate(tasks):
            if is_mtd:
                task_name = config["task_names"][t]
            else:
                task_name = task

            # Build dataloaders
            if dataset_name == "MVTEC":
                train_dataset = datasets.mvtec(train=True, unsupervised=True,
                                               replay=0, task=task, transform=None)
                test_dataset = datasets.mvtec(train=False, unsupervised=True,
                                              replay=0, task=task, transform=None)
            elif dataset_name == "MVTEC_LOCO":
                train_dataset = datasets.mvtec_loco(train=True, task=task,
                                                    replay=0, transform=None)
                test_dataset = datasets.mvtec_loco(train=False, task=task,
                                                   replay=0, transform=None)
            elif is_mtd:
                train_dataset = datasets.mtd(train=True, unsupervised=True,
                                             replay=True, transform=None,
                                             data_aug=config["data_aug"],
                                             data_aug_params=task)
                test_dataset = datasets.mtd(train=False, unsupervised=True,
                                            replay=False, transform=None,
                                            data_aug=config["data_aug"],
                                            data_aug_params=task)

            train_loader = DataLoader(train_dataset, batch_size=16,
                                      shuffle=False, collate_fn=datasets.collate)
            test_loader = DataLoader(test_dataset, batch_size=16,
                                     shuffle=False, collate_fn=datasets.collate)

            # Run evaluation
            # Note: DINOSaur.eval_one_epoch now returns a 5-tuple (added
            # gt_masks_raw_list for pixel-AUROC support in the main pipeline).
            # Ablation only uses image-level metrics, so we discard the last entry.
            patch_ad_scores, img_ad_scores, gt_masks, labels, _ = \
                model.eval_one_epoch(test_loader, neighborhood)

            # Get threshold from training data
            threshold = model.calc_percentile(train_loader, neighborhood)
            preds = (img_ad_scores > threshold).int()

            # Compute and save each metric
            for m in METRICS:
                csv_path = os.path.join(out_dir, f"{m}.csv")

                # Load or create CSV
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, index_col=0)
                else:
                    df = pd.DataFrame(columns=task_names)
                    df.index.name = "Config"

                if m == "img_acc":
                    val = ((preds == labels).sum() / len(preds)).item()
                elif m == "img_recall":
                    val = (((preds == 1) * (labels == 1)).sum() / ((labels == 1).sum())).item()
                elif m == "img_auroc":
                    val = tef.binary_auroc(img_ad_scores, labels).item()

                df.loc[row_name, task_name] = val
                df.to_csv(csv_path)

            print(f"    {task_name}: done")

            # Free eval tensors and datasets between tasks
            del train_dataset, test_dataset, train_loader, test_loader
            del patch_ad_scores, img_ad_scores, gt_masks, labels
            del threshold, preds
            cleanup_gpu()

    del model
    cleanup_gpu()


# ============================================================
# Inference profiling (edge devices, -i flag)
# ============================================================
def inference_ablation(dataset_for_weights: str = "MVTEC"):
    """
    Profile inference time and storage for each (coreset_pct, neighborhood) combo.
    Loads ablation weights trained on the specified dataset.

    Designed to run from project root on edge devices.
    """
    N_WARMUP = 5
    N_RUNS = 30

    print("=" * 70)
    print("DINOSaur Ablation — Edge Inference Profiling")
    print("=" * 70)

    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "Apple MPS"
    print(f"Device: {device_name}")
    print(f"Warmup: {N_WARMUP} | Timed runs: {N_RUNS}")
    print(f"Using weights from: {dataset_for_weights}\n")

    results = []

    for coreset_pct in CORESET_PCTS:
        wp = weight_path(dataset_for_weights, coreset_pct)
        if not os.path.exists(wp):
            print(f"  SKIP coreset_pct={coreset_pct}: weights not found at {wp}")
            continue

        print(f"--- coreset_pct={coreset_pct} ---")
        print(f"  Loading: {wp}")

        try:
            model = DINOSaur_Model(coreset_pct=coreset_pct)
            model.load(wp)
            model.eval()
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            cleanup_gpu()
            continue

        device = next(model.parameters()).device
        use_cuda = (device.type == "cuda")

        # Count params (same for all neighborhoods)
        total_params = sum(p.numel() for p in model.parameters()) / 1e6

        # Measure per-task storage
        total_storage = 0
        num_tasks = len(model.patch_memory)
        for task_key in model.patch_memory:
            total_storage += model.cls_memory[task_key].nelement() * model.cls_memory[task_key].element_size()
            total_storage += model.patch_memory[task_key].nelement() * model.patch_memory[task_key].element_size()
        per_task_MB = (total_storage / max(num_tasks, 1)) / (1024 * 1024)
        total_storage_MB = total_storage / (1024 * 1024)

        print(f"  Params: {total_params:.2f}M | {num_tasks} tasks | "
              f"{total_storage_MB:.2f} MB total ({per_task_MB:.2f} MB/task)")

        # Dummy input
        img = torch.randn(3, 224, 224, device=device)

        for neighborhood in NEIGHBORHOODS:
            config_name = f"c{fmt_coreset(coreset_pct)}_n{neighborhood}"

            # Warmup
            for _ in range(N_WARMUP):
                _ = model.predict(img.unsqueeze(0), neighborhood=neighborhood)
                if use_cuda:
                    torch.cuda.synchronize()

            # Timed runs
            times = []
            for _ in range(N_RUNS):
                if use_cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model.predict(img.unsqueeze(0), neighborhood=neighborhood)
                if use_cuda:
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)

            mean_ms = np.mean(times)
            std_ms = np.std(times)

            print(f"  n={neighborhood}: {mean_ms:.2f} +/- {std_ms:.2f} ms")

            results.append({
                "config": config_name,
                "coreset_pct": coreset_pct,
                "neighborhood": neighborhood,
                "total_params_M": round(total_params, 2),
                "per_task_storage_MB": round(per_task_MB, 2),
                "total_storage_MB": round(total_storage_MB, 2),
                "num_tasks": num_tasks,
                "inference_mean_ms": round(mean_ms, 2),
                "inference_std_ms": round(std_ms, 2),
            })

        del model
        cleanup_gpu()
        print()

    # Save results
    os.makedirs(ABLATION_RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(ABLATION_RESULTS_DIR, "edge_inference.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Summary table
    print(f"\n{'Config':<15} {'Params(M)':>10} {'MB/task':>10} {'Inference(ms)':>18}")
    print("-" * 55)
    for r in results:
        inf_str = f"{r['inference_mean_ms']:.2f} +/- {r['inference_std_ms']:.2f}"
        print(f"{r['config']:<15} {r['total_params_M']:>10.2f} {r['per_task_storage_MB']:>10.2f} {inf_str:>18}")


# ============================================================
# Aggregation: summarise per-task CSVs into coreset × neighborhood tables
# ============================================================
def aggregate_ablation(datasets: list = None):
    """
    For each dataset and metric, read the per-coreset CSV files and produce
    a summary CSV where rows = coreset_pct, columns = neighborhood, and
    each cell is the mean across tasks.

    Output:
        results/ablation/{dataset}/{dataset}_{metric}.csv
        e.g. results/ablation/MVTEC/MVTEC_img_auroc.csv

    Each summary CSV looks like:
                n0      n1      n2      n3      n5      n7
        c1    0.812   0.834   ...
        c2    0.823   0.841   ...
        ...
    """
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())

    for dataset_name in datasets:
        dataset_dir = os.path.join(ABLATION_RESULTS_DIR, dataset_name)
        if not os.path.isdir(dataset_dir):
            print(f"  SKIP aggregate: {dataset_dir} not found")
            continue

        print(f"\n--- Aggregating: {dataset_name} ---")

        for metric in METRICS:
            # Collect rows from all per-coreset subdirectories
            summary = {}  # {coreset_tag: {neighborhood: mean_value}}

            for coreset_pct in CORESET_PCTS:
                coreset_tag = f"c{fmt_coreset(coreset_pct)}"
                csv_path = os.path.join(dataset_dir, coreset_tag, f"{metric}.csv")

                if not os.path.exists(csv_path):
                    continue

                df = pd.read_csv(csv_path, index_col=0)

                for neighborhood in NEIGHBORHOODS:
                    row_name = f"{coreset_tag}_n{neighborhood}"
                    if row_name not in df.index:
                        continue

                    row_vals = pd.to_numeric(df.loc[row_name], errors='coerce')
                    mean_val = round(row_vals.mean(), 4)
                    summary.setdefault(coreset_tag, {})[f"n{neighborhood}"] = mean_val

            if not summary:
                print(f"  {metric}: no data found")
                continue

            # Build summary DataFrame: rows = coreset tags, columns = neighborhoods
            neighborhood_cols = [f"n{n}" for n in NEIGHBORHOODS]
            summary_df = pd.DataFrame.from_dict(summary, orient='index',
                                                 columns=neighborhood_cols)
            summary_df.index.name = "coreset"

            # Sort rows by coreset percentage (numeric order)
            def coreset_sort_key(tag):
                return int(tag[1:])  # "c5" -> 5, "c10" -> 10
            summary_df = summary_df.loc[sorted(summary_df.index, key=coreset_sort_key)]

            out_path = os.path.join(dataset_dir, f"{dataset_name}_{metric}.csv")
            summary_df.to_csv(out_path)
            print(f"  Saved: {out_path}")
            print(summary_df.to_string())
            print()

    # --- Global aggregation: average across all datasets for each metric ---
    print(f"\n{'='*60}")
    print("Global aggregation: averaging across datasets")
    print(f"{'='*60}")

    for metric in METRICS:
        dataset_dfs = []
        for dataset_name in datasets:
            csv_path = os.path.join(ABLATION_RESULTS_DIR, dataset_name,
                                    f"{dataset_name}_{metric}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, index_col=0)
            dataset_dfs.append(df)

        if not dataset_dfs:
            print(f"  {metric}: no per-dataset summaries found")
            continue

        # Average all dataset tables: align on shared (coreset, neighborhood) pairs
        combined = dataset_dfs[0].copy()
        for df in dataset_dfs[1:]:
            combined = combined.add(df, fill_value=0)
        global_df = (combined / len(dataset_dfs)).round(4)

        # Sort rows by coreset percentage
        def coreset_sort_key(tag):
            return int(tag[1:])
        global_df = global_df.loc[sorted(global_df.index, key=coreset_sort_key)]

        out_path = os.path.join(ABLATION_RESULTS_DIR, f"{metric}.csv")
        global_df.to_csv(out_path)
        print(f"\n  Saved: {out_path}  ({len(dataset_dfs)} datasets averaged)")
        print(global_df.to_string())
        print()


# ============================================================
# Main
# ============================================================
def main():
    global CORESET_PCTS, NEIGHBORHOODS

    parser = argparse.ArgumentParser(description="DINOSaur ablation studies")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-t", "--train", action="store_true",
                      help="Train ablation models (saves weights, no eval)")
    mode.add_argument("-e", "--evaluate", action="store_true",
                      help="Evaluate trained models at all neighborhoods (saves per-task CSVs)")
    mode.add_argument("-a", "--aggregate", action="store_true",
                      help="Aggregate per-task results into coreset x neighborhood summary CSVs")
    mode.add_argument("-i", "--inference", action="store_true",
                      help="Inference profiling mode (for edge devices)")
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Which datasets to run (default: all)")
    parser.add_argument("--coreset_pcts", nargs="+", type=float,
                        default=CORESET_PCTS,
                        help="Coreset percentages to test (default: 0.01 0.025 0.05 0.1 0.2)")
    parser.add_argument("--neighborhoods", nargs="+", type=int,
                        default=NEIGHBORHOODS,
                        help="Neighborhood sizes to test (default: 0 1 2 3 5 7)")
    parser.add_argument("--weights_dataset", type=str, default="MVTEC",
                        help="Which dataset's weights to use for -i mode (default: MVTEC)")
    args = parser.parse_args()

    CORESET_PCTS = args.coreset_pcts
    NEIGHBORHOODS = args.neighborhoods

    if args.train:
        for dataset_name in args.datasets:
            for coreset_pct in CORESET_PCTS:
                train_ablation(dataset_name, coreset_pct)
        print("\n" + "=" * 60)
        print("Training complete! Run with -e to evaluate.")
        print("=" * 60)

    elif args.evaluate:
        for dataset_name in args.datasets:
            for coreset_pct in CORESET_PCTS:
                evaluate_ablation(dataset_name, coreset_pct)
        # Auto-aggregate after eval
        aggregate_ablation(datasets=args.datasets)
        print("\n" + "=" * 60)
        print("Evaluation complete!")
        print(f"Results saved to {ABLATION_RESULTS_DIR}/")
        print("=" * 60)

    elif args.aggregate:
        aggregate_ablation(datasets=args.datasets)

    elif args.inference:
        inference_ablation(dataset_for_weights=args.weights_dataset)


if __name__ == "__main__":
    main()
