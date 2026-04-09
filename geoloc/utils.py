import os
import math
import torch
import wandb
import inspect
import numpy as np
import torchvision
from pyproj import Transformer
import torchvision.transforms.functional as TF
import datetime

from .config_parser import class_from_config, load_config

DEBUG = int(os.getenv("DEBUG", 0))

WGS84_A = 6378137.0  # semi-major axis in meters
WGS84_B = 6356752.314245  # semi-minor axis in meters
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)  # eccentricity squared


def crs_transform(point: tuple, src_crs: str, tgt_crs: str):
    transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
    return transformer.transform(point[0], point[1])

def geodetic_to_ecef(lat_deg, lon_deg, h):
    """
    Convert geodetic coordinates (lat, lon, alt) to ECEF (x, y, z)
    Returns:
            x, y, z in meters
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    X = (N + h) * math.cos(lat) * math.cos(lon)
    Y = (N + h) * math.cos(lat) * math.sin(lon)
    Z = (N * (1 - WGS84_E2) + h) * math.sin(lat)
    return X, Y, Z

def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF (x, y, z) to geodetic coordinates (lat, lon, alt)
    Returns:
            lat, lon in degrees
            alt in meters
    """
    # Longitude
    lon = np.arctan2(y, x)

    # Iterative computation for latitude
    r = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, r * (1 - WGS84_E2))  # initial guess
    lat_prev = 0
    while np.abs(lat - lat_prev) > 1e-12:
        lat_prev = lat
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        alt = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - WGS84_E2 * N / (N + alt)))

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = r / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    return lat, lon, alt

def color_text(text, color="red"):
    colors = {
        "red": "\033[91m",
        "yellow": "\033[93m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

# def load_model(config):    
#     model_config = config.model
#     if hasattr(config.model, "load_checkpoint"):
#         dirpath = os.path.dirname(config.model.load_checkpoint)
#         if os.path.exists(os.path.join(dirpath, "train_config.yaml")):
#             train_config_path = os.path.join(dirpath, "train_config.yaml")
#             model_config = load_config(train_config_path).model

#         model_cls, init_args = class_from_config(model_config, instantiate=False)
#         model = model_cls.load_from_checkpoint(
#             config.model.load_checkpoint, strict=False, weights_only=False,
#             **init_args
#         )
#     else:
#         model_cls, init_args = class_from_config(model_config, instantiate=False)
#         model = model_cls(**init_args)
#     model.my_config = model_config
#     return model

def load_model_from_my_checkpoint(checkpoint_path: str):
    dirpath = os.path.dirname(checkpoint_path)
    if os.path.exists(os.path.join(dirpath, "train_config.yaml")):
        train_config_path = os.path.join(dirpath, "train_config.yaml")
        model_config = load_config(train_config_path).model
    model_cls, init_args = class_from_config(model_config, instantiate=False)
    model = model_cls.load_from_checkpoint(
        checkpoint_path, strict=False, weights_only=False,
        **init_args
    )
    model.my_config = model_config
    return model

def load_model(config=None, checkpoint_path: str = None, do_compile=True):
    assert (config is not None) != (checkpoint_path is not None), \
        "Exactly one of config or checkpoint_path must be provided"
    if checkpoint_path is not None:
        return load_model_from_my_checkpoint(checkpoint_path)
    model = class_from_config(config.model)
    if do_compile and hasattr(model, "compile"):
        model.compile()
    model.my_config = config.model
    return model

def get_model_type(model) -> str:
    return model.__class__.__name__

def get_dataset_type(dataset) -> str:
    return dataset.__module__.split(".")[-1].lower()

def tensor2wandbimg(tensor, max_imgs=4, nrow=4):
    assert len(tensor.shape) == 4, "Input tensor must be 4D (B, C, H, W)"
    nrow = min(tensor.shape[0], nrow)

    # Force resize to avoid bus errors
    tensor = TF.resize(tensor, 224)
    
    return wandb.Image(
        TF.to_pil_image(
            torchvision.utils.make_grid(
                tensor[:max_imgs, ...].detach().cpu(),
                normalize=True, scale_each=True, nrow=nrow)))

def create_run_name(config):
    run_name = ""

    if hasattr(config, "model"):
        model_cls = config.model.class_path.split(".")[-1].lower()
        run_name += f"{model_cls}_"

        if hasattr(config.model.init_args, "backbone"):
            for k in ["model_name", "pretrained_model_name_or_path"]:
                backbone_name = getattr(
                    config.model.init_args.backbone.init_args, k, None)
                if backbone_name is not None:
                    backbone_name = backbone_name.replace("/", "-")
                    run_name += f"bb:{backbone_name}_"
                    break
        if hasattr(config.model.init_args, "aggregator"):
            aggregator_name = config.model.init_args.aggregator.class_path.split(".")[-1].lower()
            run_name += f"agg:{aggregator_name}_"

        if hasattr(config.model.init_args, "optimizer"):
            run_name += f"lr:{config.model.init_args.optimizer.init_args.lr:.0e}_"

        if hasattr(config.model.init_args, "scheduler"):
            scheduler_name = config.model.init_args.scheduler.class_path.split(".")[-1].lower()
            run_name += f"sch:{scheduler_name}_"

        if hasattr(config.model.init_args, "loss"):
            loss_name = config.model.init_args.loss.class_path.split(".")[-1].lower()
            run_name += f"loss:{loss_name}_"

        if hasattr(config.model.init_args, "st_loss"):
            st_loss_name = config.model.init_args.st_loss.class_path.split(".")[-1].lower()
            run_name += f"stloss:{st_loss_name}_"

    if hasattr(config, "dataset"):
        dataset_name = f"{config.dataset.class_path.split('.')[-1]}"
        run_name += f"ds:{dataset_name}_"
        if getattr(config.dataset.init_args, "north_align", False):
            run_name += f"north-align_"
    
    if hasattr(config, "trainer"):
        if hasattr(config.trainer, "max_epochs"):
            run_name += f"ep:{config.trainer.max_epochs}_"
        if hasattr(config.trainer, "precision"):
            run_name += f"prec:{config.trainer.precision}_"

    run_name += f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return run_name

def requires_arg(func, arg_name):
    sig = inspect.signature(func)
    param = sig.parameters.get(arg_name)
    
    if param is None:
        return False
    
    return (
        param.default is inspect._empty
        and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    )

def hasarg(func, arg_name):
    sig = inspect.signature(func)
    return arg_name in sig.parameters