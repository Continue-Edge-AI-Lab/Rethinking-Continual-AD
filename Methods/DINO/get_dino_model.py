import torch
from pathlib import Path

def get_dino_model(model_variant: str, model_weights_dir: str = "Methods/DINO/dinov3_weights"):
    """
    Each of these models are the standard models trained on LVD-1690M.
    Possible model_variant strings are:
        - dinov3_vits16
        - dinov3_vits16plus
        - dinov3_vitb16
        - dinov3_vitl16

        - dinov3_convnext_tiny
        - dinov3_convnext_small
        - dinov3_convnext_base
        - dinov3_convnext_large

    :param model_variant: a string, detailing which variant of DINOv3 to use
    :param model_weights_dir: a string, detailing where to load the weights from
    :return: dino_model: a torch model
    """
    available_models = {
        # Commented out models weren't tested because they're too large

        # ViT-based models
        'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
        'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        # 'dinov3_vith16plus'
        # 'dinov3_vit7b16',

        # ConvNeXt-based models
        'dinov3_convnext_tiny': 'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth',
        'dinov3_convnext_small': 'dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth',
        'dinov3_convnext_base': 'dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth',
        'dinov3_convnext_large': 'dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth'
    }

    weights_loc = str(Path(model_weights_dir).joinpath(available_models[model_variant]))
    dino_model = torch.hub.load("Methods/DINO/dinov3",
                                model_variant,
                                source='local',
                                weights=weights_loc)

    return dino_model