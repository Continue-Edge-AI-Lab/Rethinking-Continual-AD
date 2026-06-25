"""
Main function to run on the GPU Cluster.

Used tmux for creating a session that won't stop if the SSH
connection is interrupted.

- To start session:
>>> tmux new -s chadw (<- session name)
- To list sessions:
>>> tmux ls
- To reattach to a session:
>>> tmux a -t chadw

To run the whole program, make sure to create an error.log file with:
>>> touch error.log
Then when running the program, redirect stderr to that log file like so:
>>> python main.py 2> error.log
"""

from utils.train import train_model
from utils.eval import eval_model
import torch

def main():
    # Hyperparameters for training/testing
    torch.manual_seed(42)
    # Whether to do training, with which models, and on which datasets
    TRAIN = True
    EVAL = False
    models = {
        "DNE":False,
        "IUF":False,
        "UCAD":True,
        "Patchcore":True,
        "EfficientAD":True,
        "DINOSaur":True
    }
    datasets = {
        "MVTEC":False,
        "MVTEC_LOCO":False,
        "MTD":True
    }
    NUM_EPOCHS = 100
    BATCH_SIZE = 12
    LEARNING_RATE = 0.00075
    WEIGHT_DECAY = 0.0001
    EVAL_METRICS = [
        # "img_acc",
        # "img_recall",
        # "img_auroc",
        "pixel_auroc"  # Added for ECCV 2026 rebuttal — pooled pixel-level AUROC.
                       # DNE skips this (no spatial information available).
    ]

    data_aug = {
        "color": [
            [0.0, 0.05],
            [0.05, 0.1],
            [0.1, 0.15],
            [0.15, 0.2],
            [0.2, 0.25],
            [0.25, 0.3],
            [0.3, 0.35],
            [0.35, 0.4],
            [0.4, 0.45],
            [0.45, 0.5]
        ],
        "blur": [
            [1, 0.5],
            [3, 1],
            [5, 1.5],
            [7, 2],
            [9, 2.5],
            [11, 3],
            [13, 3.5],
            [15, 4],
            [17, 4.5],
            [19, 5],
        ],
        "geometric": [
            [2, 1, 0.01, 1],
            [4, 2, 0.02, 2],
            [6, 3, 0.03, 3],
            [8, 4, 0.04, 4],
            [10, 5, 0.05, 5],
            [12, 6, 0.06, 6],
            [14, 7, 0.07, 7],
            [16, 8, 0.08, 8],
            [18, 9, 0.09, 9],
            [20, 10, 0.10, 10]
        ]
    }
    # Running Training Experiments
    if TRAIN:
        for model in models.keys():
            if models[model]:
                print(f"Starting training for model: {model}...")
                if datasets['MVTEC']:
                    train_model(model_type=model,
                                dataset='MVTEC',
                                num_epochs=NUM_EPOCHS,
                                batch_size=BATCH_SIZE,
                                criterion=torch.nn.CrossEntropyLoss(),
                                learning_rate=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY
                                )
                if datasets['MVTEC_LOCO']:
                    train_model(model_type=model,
                                dataset='MVTEC_LOCO',
                                num_epochs=NUM_EPOCHS,
                                batch_size=BATCH_SIZE,
                                criterion=torch.nn.CrossEntropyLoss(),
                                learning_rate=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY
                                )
                if datasets['MTD']:
                    for distortion in data_aug.keys():
                        train_model(model_type=model,
                                    dataset='MTD',
                                    num_epochs=NUM_EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    criterion=torch.nn.CrossEntropyLoss(),
                                    learning_rate=LEARNING_RATE,
                                    tasks=data_aug[distortion],
                                    data_aug=distortion
                                    )

    # Running Evaluation Experiments
    if EVAL:
        for model in models.keys():
            if models[model]:
                for dataset in datasets.keys():
                    if datasets[dataset]:
                        out = eval_model(model_type=model,
                                         batch_size=BATCH_SIZE,
                                         data_aug=data_aug,
                                         dataset=dataset,
                                         metrics=EVAL_METRICS
                                         )

    return

if __name__ == "__main__":
    main()