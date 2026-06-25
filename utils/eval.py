import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from IPython.display import clear_output

from Methods.DINO.DINOSaur import DINOSaur_Model
from Methods.DNE.dne import DNE_Model
from Methods.EfficientAD.efficient_ad import EfficientAD_Model
from Methods.IUF.iuf import IUF_Model, IUF_Loss
from Methods.UCAD.ucad import UCAD_Model, UCAD_Contrastive_loss
from Methods.PatchCore.patchcore import Patchcore_Model
import datasets

def eval_model(model_type: str,
                batch_size: int,
               dataset: str,
               metrics: list[str],
                **kwargs):
    """
    Test a model on all experiments. Assumes it's running from the root folder,
    and saving of plots and models assumes this current directory structure.
    Args:
        model_type: a string of which model to train ("DNE", "IUF", "UCAD")
        batch_size: an int of how many samples per batch to train the model
        dataset: a string of which dataset to train on
        metrics: a list of strings of which metrics to use for evaluating the model
        kwargs: additional arguments. Current assumed ones are:
            - data_aug, for MTD, which should contain a dict of the type of data augmentation for MTD,
                        with keys being any combination of ["color", "blur", "geometric"]
                        and each entry being a list of lists, where each individual list
                        is a set of data augmentation parameters

    Returns:
    """
    # kwargs takes any other parameters not explicitly stated above, and turns them
    # into a dict(). kwargs.get() allows us to search that dict, or if it doesn't exist,
    # provide a default option

    # Go through each dataset
    if dataset in ['MVTEC', 'MVTEC_V2', 'MVTEC_LOCO']:
        if dataset == 'MVTEC':
            tasks = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                     'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                     'transistor', 'wood', 'zipper']
        elif dataset == "MVTEC_V2":
            tasks = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal',
                     'vial', 'wallplugs', 'walnuts']
        elif dataset == "MVTEC_LOCO":
            tasks = ['breakfast_box', 'screw_bag', 'pushpins',
                     'splicing_connectors', 'juice_bottle']


        # Iterate through each unsupervised/supervised experiment
        for unsupervised in [True, False]:
            # Load in model
            # Get model
            match model_type:
                case "DNE":
                    model = DNE_Model()
                    final_model = DNE_Model()
                case "IUF":
                    if unsupervised:
                        model = IUF_Model(num_tasks=len(tasks))
                        final_model = IUF_Model(num_tasks=len(tasks))
                    else:
                        continue
                case "UCAD":
                    if unsupervised:
                        model = UCAD_Model()
                        final_model = UCAD_Model()
                    else:
                        continue
                case "Patchcore":
                    if unsupervised:
                        model = Patchcore_Model()
                        final_model = Patchcore_Model()
                    else:
                        continue
                case "EfficientAD":
                    if unsupervised:
                        model = EfficientAD_Model()
                        final_model = EfficientAD_Model()
                    else:
                        continue
                case "DINOSaur":
                    if unsupervised:
                        model = DINOSaur_Model()
                        final_model = DINOSaur_Model()
                    else:
                        continue

            # Load in final model
            final_model.load(f"./models/{model_type}/{model_type}_{dataset}_{"unsupervised" if unsupervised else "supervised"}_final_weights.pth")
            if model_type == "DNE":
                final_model.generate_global_dist()

            # Iterate through tasks
            for t in range(len(tasks)):
                # Get task
                task = tasks[t]
                # Get task-specific model
                dir_name = f"./models/{model_type}/{dataset}"
                # Get model-specific transform, if needed
                if hasattr(model, 'transform'):
                    model_transform = model.transform
                else:
                    model_transform = None
                if t < (len(tasks) - 1):
                    model.load(f"{dir_name}/{task}_{"unsupervised" if unsupervised else "supervised"}_weights.pth")
                    # If DNE, we need to generate our global distribution for inference
                    if model_type == "DNE":
                        model.generate_global_dist()

                clear_output(wait=False)
                # Print status values
                print(f"Testing {model_type}-{dataset}-{'Unsupervised' if unsupervised else 'Supervised'}",
                      "--------------------",
                      f"Current Task: {task}",
                      sep="\n")

                # Run through current task
                if dataset == "MVTEC":
                    test_dataset = datasets.mvtec(train=False, task=task, unsupervised=unsupervised,
                                                  transform=model_transform)
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                                 shuffle=False, collate_fn=datasets.collate)
                    # Training data used for thresholding pixel classifications
                    train_dataset=datasets.mvtec(train=True, task=task, unsupervised=unsupervised,
                                                 transform=model_transform)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=False, collate_fn=datasets.collate)
                elif dataset == "MVTEC_V2":
                    test_dataset = datasets.mvtec_v2(train=False, task=task,
                                                  transform=model_transform)
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                                 shuffle=False, collate_fn=datasets.collate)
                    # Training data used for thresholding pixel classifications
                    train_dataset = datasets.mvtec_v2(train=True, task=task,
                                                 transform=model_transform)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=False, collate_fn=datasets.collate)

                elif dataset == "MVTEC_LOCO":
                    test_dataset = datasets.mvtec_loco(train=False, task=task,
                                                     transform=model_transform)
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                                 shuffle=False, collate_fn=datasets.collate)
                    # Training data used for thresholding pixel classifications
                    train_dataset = datasets.mvtec_loco(train=True, task=task,
                                                      transform=model_transform)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                                  shuffle=False, collate_fn=datasets.collate)

                # Evaluate one epoch of the test_dataset
                if t < (len(tasks) - 1):
                    model.calc_results(test_dataloader,
                                       dataset,
                                      task,
                                      tasks,
                                      "unsupervised" if unsupervised else "supervised",
                                      metrics,
                                        final=False,
                                      train_dataloader=train_dataloader)
                final_model.calc_results(test_dataloader,
                                         dataset,
                                         task,
                                         tasks,
                                         "unsupervised" if unsupervised else "supervised",
                                         metrics,
                                         final=True,
                                         train_dataloader=train_dataloader)
    elif dataset == "MTD":
        task_dict = kwargs.get('data_aug')
        for distortion in task_dict.keys():
            tasks = [] # list of tuples, containing tuples of (distortion, data_aug_params)
            task_names = []
            for task in task_dict[distortion]:
                task_name = distortion + "_"
                tasks.append((distortion, task))
                for param in task:
                    task_name += (str(param).replace(".", ""))
                    if param != task[-1]:
                        task_name += "_"
                task_names.append(task_name)

            # Iterate through each unsupervised/supervised experiment
            for unsupervised in [True, False]:
                # Load in model
                # Get model
                match model_type:
                    case "DNE":
                        model = DNE_Model()
                        final_model = DNE_Model()
                    case "IUF":
                        if unsupervised:
                            model = IUF_Model(num_tasks=len(tasks))
                            final_model = IUF_Model(num_tasks=len(tasks))
                        else:
                            continue
                    case "UCAD":
                        if unsupervised:
                            model = UCAD_Model()
                            final_model = UCAD_Model()
                        else:
                            continue
                    case "Patchcore":
                        if unsupervised:
                            model = Patchcore_Model()
                            final_model = Patchcore_Model()
                        else:
                            continue
                    case "EfficientAD":
                        if unsupervised:
                            model = EfficientAD_Model()
                            final_model = EfficientAD_Model()
                        else:
                            continue
                    case "DINOSaur":
                        if unsupervised:
                            model = DINOSaur_Model()
                            final_model = DINOSaur_Model()
                        else:
                            continue

                # Load in final model
                final_model.load(f"./models/{model_type}/{model_type}_MTD_{distortion}_{"unsupervised" if unsupervised else "supervised"}_final_weights.pth")
                if model_type == "DNE":
                    final_model.generate_global_dist()

                # Iterate through tasks
                for t in range(len(tasks)):
                    # Get task
                    task = tasks[t]
                    # Get task name, useful for saving data
                    task_name = task_names[t]
                    # Get model-specific transform, if needed
                    if hasattr(model, 'transform'):
                        model_transform = model.transform
                    else:
                        model_transform = None
                    # Get task-specific model
                    dir_name = f"./models/{model_type}/{'MTD_'+distortion}"
                    if t < (len(tasks) - 1):
                        model.load(f"{dir_name}/{task_name}_{"unsupervised" if unsupervised else "supervised"}_weights.pth")
                        # If DNE, we need to generate our global distribution for inference
                        if model_type == "DNE":
                            model.generate_global_dist()


                    clear_output(wait=False)
                    # Print status values
                    print(f"Testing {model_type}-MTD-{distortion}-{'Unsupervised' if unsupervised else 'Supervised'}",
                          "--------------------",
                          f"Current Task: {task}",
                          sep="\n")

                    # Run through current and all previous tasks
                    test_dataset = datasets.mtd(train=False, unsupervised=unsupervised, transform=model_transform,
                                                    data_aug=task[0], data_aug_params=task[1])
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                                 shuffle=False, collate_fn=datasets.collate)
                    # Training data used for thresholding
                    train_dataset=datasets.mtd(train=True, unsupervised=unsupervised, transform=model_transform,
                                               data_aug=task[0], data_aug_params=task[1])
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=False, collate_fn=datasets.collate)
                    # Evaluate one epoch of the test_dataset
                    if t < (len(tasks) - 1):
                        model.calc_results(test_dataloader,
                                          "MTD",
                                          task_name,
                                          task_names,
                                          "unsupervised" if unsupervised else "supervised",
                                          metrics,
                                            final=False,
                                          train_dataloader=train_dataloader)
                    final_model.calc_results(test_dataloader,
                                       "MTD",
                                       task_name,
                                       task_names,
                                       "unsupervised" if unsupervised else "supervised",
                                       metrics,
                                       final=True,
                                       train_dataloader=train_dataloader)

    return