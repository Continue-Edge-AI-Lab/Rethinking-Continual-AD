import time
import os
import json
import gc
import torch
from torch.utils.data import DataLoader
from IPython.display import clear_output
from Methods.DNE.dne import DNE_Model
from Methods.IUF.iuf import IUF_Model, IUF_Loss
from Methods.UCAD.ucad import UCAD_Model, UCAD_Contrastive_loss
from Methods.PatchCore.patchcore import Patchcore_Model
from Methods.EfficientAD.efficient_ad import EfficientAD_Model
from Methods.DINO.DINOSaur import DINOSaur_Model
import datasets


class EarlyStopper:
    """Stop training once the loss stops changing in absolute terms.

    Measures the absolute change in loss between consecutive epochs. After
    `patience` consecutive epochs whose absolute change is below `min_delta`,
    check() returns True. This tolerates noisy losses that still jump by several
    units while trending (e.g. -31 -> -26 -> -29) and stops only on a true plateau.
    """
    def __init__(self, min_delta=0.5, patience=3):
        self.min_delta = min_delta    # absolute loss change below which an epoch counts as "no progress"
        self.patience = patience      # consecutive flat epochs allowed before stopping
        self.prev_loss = None
        self.counter = 0

    def check(self, loss) -> bool:
        # Returns True if training should STOP.
        if self.prev_loss is None:              # first epoch: nothing to compare yet
            self.prev_loss = loss
            return False
        change = abs(loss - self.prev_loss)     # absolute change from the previous epoch
        self.prev_loss = loss
        if change < self.min_delta:             # barely moved this epoch
            self.counter += 1
        else:                                   # still learning -> reset
            self.counter = 0
        return self.counter >= self.patience


def train_model(model_type: str,
                dataset: str,
                num_epochs: int,
                batch_size: int,
                **kwargs):
    """
    Train a model with given criterion and optimizer. Assumes it's running from the root folder,
    and saving of plots and models assumes this current directory structure.
    Args:
        model_type: a string of which model to train ("DNE", "IUF", "UCAD")
        dataset: a string of which dataset to use ("MVTEC", "MTD")
        num_epochs: an int of how many epochs to train the model
        batch_size: an int of how many samples per batch to train the model
        kwargs: additional arguments. Current assumed ones are:
            - criterion
            - learning_rate
            - weight_decay
            - tasks, if using MTD, which should contain a list of the dataset augmentation windows
                    (will be passed to datasets.mtd() as data_aug_params)
            - data_aug, if using MTD, which should contain a string of the type of data augmentation for MTD,
                        which should be a string in ["color", "blur", "geometric"]
    Returns:
    """

    # kwargs takes any other parameters not explicitly stated above, and turns them
    # into a dict(). kwargs.get() allows us to search that dict, or if it doesn't exist,
    # provide a default option
    criterion = kwargs.get('criterion', torch.nn.CrossEntropyLoss())

    # Set up tasks
    if dataset == "MVTEC":
        tasks = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                       'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                       'transistor', 'wood', 'zipper']
    elif dataset == "MVTEC_V2":
        tasks = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal',
                 'vial', 'wallplugs', 'walnuts']
    elif dataset == "MVTEC_LOCO":
        tasks = ['breakfast_box', 'screw_bag', 'pushpins',
                 'splicing_connectors', 'juice_bottle']
    elif dataset == "MTD":
        tasks = kwargs.get('tasks')
        data_aug = kwargs.get('data_aug')
        assert data_aug in ["color", "blur", "geometric"]

    # Tracks losses across each experiment, then task.
    # Contains keys of: train_task_losses[unsupervised][task][epoch]
    train_task_losses = {}

    # Iterate through each unsupervised/supervised experiment

    for unsupervised in [True, False]:
        replay = False  # will only use it for patchcore and EfficientAD
        # A new model for each unsupervised/supervised task
        match model_type:
            case "DNE":
                model = DNE_Model()
            case "IUF":
                if unsupervised:
                    model = IUF_Model(num_tasks=10 if dataset=="MTD" else len(tasks))
                else:
                    continue
            case "UCAD":
                if unsupervised:
                    model = UCAD_Model(vit_output_layer=5)
                else:
                    continue
            # Basic AD methods
            case "Patchcore":
                if unsupervised:
                    model = Patchcore_Model(backbone='resnet18')
                    replay = True
                else:
                    continue
            case "EfficientAD":
                if unsupervised:
                    model = EfficientAD_Model()
                    replay = True
                else:
                    continue
            case "DINOSaur":
                if unsupervised:
                    model = DINOSaur_Model(coreset_pct=0.10)
                else:
                    continue

        optimizer = torch.optim.Adam(model.parameters(),
                                         lr=kwargs.get('learning_rate', 0.001),
                                     weight_decay=kwargs.get('weight_decay', 0.0001))

        # Keeps track of current experiment losses (unsupervised only)
        exp_losses = {}
        # Iterate through tasks
        for t in range(len(tasks)):
            task = tasks[t]
            if hasattr(model, 'transform'):
                model_transform = model.transform
            else:
                model_transform = None

            # Get data
            if dataset == "MVTEC":
                task_dataset = datasets.mvtec(train=True,
                                              unsupervised=True,
                                              replay=100 if replay else 0,
                                              task=task,
                                              transform=model_transform)
                task_name = task
            elif dataset == "MVTEC_V2":
                task_dataset = datasets.mvtec_v2(train=True,
                                                 task=task,
                                                 transform=model_transform)
                task_name = task
            elif dataset == "MVTEC_LOCO":
                task_dataset = datasets.mvtec_loco(train=True,
                                                 task=task,
                                                   replay=100 if replay else 0,
                                                 transform=model_transform)
                task_name = task
            elif dataset == "MTD":
                task_dataset = datasets.mtd(train=True,
                                            unsupervised=True,
                                            replay=True,
                                            transform=model_transform,
                                            data_aug=data_aug,
                                            data_aug_params=task)
                task_name = f"{data_aug}_"
                for param in task:
                    task_name += str(param).replace(".", "")
                    if param != task[-1]:
                        task_name += "_"

            dataloader = DataLoader(task_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=datasets.collate)

            if model_type == 'EfficientAD':
                model.calc_teacher_params(task_dataset)

            task_loss = []
            # Early stopping: stop once the loss improves by < rel_tol (0.1%) for `patience` epochs.
            stopper = EarlyStopper(min_delta=0.5, patience=3)
            # Run through each epoch
            for e in range(num_epochs):
                start_time = time.time()
                clear_output(wait=False)
                print(f"Running {'unsupervised' if unsupervised else 'supervised'} training on {dataset} ({task_name}):")
                print(f"Epoch {e+1}/{num_epochs}")
                if e>2:
                    print("----------------------")
                    print('Previous Epoch Losses: ')
                    for i in [-3, -2, -1]:
                        print(f"{task_loss[i]}")
                    print("----------------------")
                    print("Last epoch time:")
                    mins = int(curr_epoch_time / 60)
                    secs = int(curr_epoch_time % 60)
                    print(f"{mins} minutes, {secs} seconds")

                loss = model.train_one_epoch(dataset=task_dataset,
                                            dataloader=dataloader,
                                             task_name=task_name,
                                             optimizer=optimizer,
                                             criterion=criterion,
                                             task_num=(t+1),
                                             update_z_epoch=True if (e+1)==num_epochs else False,
                                             final_epoch=True if (e+1)==num_epochs else False,)

                task_loss.append(loss)
                curr_epoch_time = time.time() - start_time

                # Patchcore only needs one epoch for training, as it uses the nominal samples
                if model_type in ['Patchcore', 'DINOSaur']:
                    break

                # Early stopping for methods that train over multiple epochs.
                is_last_allowed = (e + 1 == num_epochs)
                if not is_last_allowed and stopper.check(loss):
                    # DNE only accumulates its per-task distribution when update_z_epoch=True,
                    # so run one finalization pass before breaking or update_memory() saves nothing.
                    if model_type == "DNE":
                        model.train_one_epoch(dataset=task_dataset, dataloader=dataloader,
                                              task_name=task_name, optimizer=optimizer,
                                              criterion=criterion, task_num=(t + 1),
                                              update_z_epoch=True, final_epoch=True)
                    print(f"Early stopping at epoch {e+1}/{num_epochs} "
                          f"(< {stopper.min_delta} loss change for {stopper.patience} epochs)")
                    break

            # Update current experiment loss with current task_loss list
            exp_losses[task_name] = task_loss

            # Save memory distribution for this task in self.memory
            if model_type == "DNE":
                model.update_memory()
            if model_type == "IUF":
                model.save_task_state()
            if model_type == "UCAD":
                model.update_memory(dataloader)

            # Save task-specific params
            if t == (len(tasks) - 1):
                # Save final model
                if dataset == "MTD":
                    model.save(f"./models/{model_type}/{model_type}_MTD_{data_aug}_{"unsupervised" if unsupervised else "supervised"}_final_weights.pth")
                else:
                    model.save(f"./models/{model_type}/{model_type}_{dataset}_{"unsupervised" if unsupervised else "supervised"}_final_weights.pth")
            else:
                # Make sure directories exist
                dataset_dir_name = f"MTD_{data_aug}" if dataset == "MTD" else dataset
                dir_name = f"./models/{model_type}/{dataset_dir_name}"
                os.makedirs(f"{dir_name}/", exist_ok=True)
                # Save task-specific model
                model.save(f"{dir_name}/{task_name}_{"unsupervised" if unsupervised else "supervised"}_weights.pth")

            # Free this task's dataloader/dataset and force collection before the next
            # task so CPU/GPU memory does not accumulate or overlap across tasks.
            del dataloader, task_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        exp = "unsupervised" if unsupervised else "supervised"
        train_task_losses[exp] = exp_losses

        # Save training results as a JSON
        if dataset == "MTD":
            filename = f"{model_type}_{data_aug}_MTD.json"
        else:
            filename = f"{model_type}_{dataset}.json"

        filepath = f"results/training_loss/{filename}"
        with open(filepath, "w") as f:
            json.dump(train_task_losses, f, indent=2)

    return