# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python geoloc/pipeline/train_or_fit.py --config configs/contrastive/VPAir/contrastive_multisimilarity.yaml

import os
import argparse
import torch
from torch.utils.data import DataLoader
from geoloc.data.utils import collate_with_geometry
from geoloc.trainer import GeolocTrainer

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from geoloc.config_parser import load_config, save_config, namespace_to_dict, class_from_config, func_from_string
from geoloc.utils import load_model, create_run_name

from geoloc.data.datasets.BigBoy import BigBoyTrainDataset, bigboy_collate_fn

def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser


def train(config):

    # torch.set_float32_matmul_precision('medium')

    train_collate_func = collate_with_geometry
    if hasattr(config.dataloader, "collate_fn") and config.dataloader.collate_fn is not None:
        train_collate_func = func_from_string(config.dataloader.pop("collate_fn"))

    train_dataset = class_from_config(config.dataset)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=train_collate_func,
        **vars(config.dataloader)
    )
    build_dataloader = None
    if hasattr(config, "build_config") and config.build_config is not None:
        build_dataset = class_from_config(config.build_config.dataset)
        build_dataloader = DataLoader(
            build_dataset,
            shuffle=False,
            batch_size=config.dataloader.batch_size,
            drop_last=False,
            collate_fn=collate_with_geometry,
        )
    benchmark_dataloader = None
    if hasattr(config, "benchmark_config") and config.benchmark_config is not None:
        benchmark_dataset = class_from_config(config.benchmark_config.dataset)
        benchmark_dataloader = DataLoader(
            benchmark_dataset,
            shuffle=False,
            batch_size=1,
            drop_last=False,
            collate_fn=collate_with_geometry,
        )

    model = load_model(config, do_compile=True)
    model_cls_name = model.__class__.__name__.lower()

    dataset_name = config.dataset.class_path.split(".")[-2]
    run_name = create_run_name(config)
    logger = WandbLogger(
        project="geoloc",
        group=model_cls_name,
        name=run_name,
        log_model=False,
        config=namespace_to_dict(config),
        **vars(config.logger)
    )
    
    savedir = f"/home/grega/geoloc/logs/checkpoints/{model_cls_name}/{dataset_name}/{run_name}"
    os.makedirs(savedir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=savedir,
        filename="{epoch:02d}-{step:06d}",
        save_last=True,
        every_n_epochs=config.trainer.pop("save_checkpoint_every_n_epochs", 20),
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = GeolocTrainer(
        **vars(config.trainer), 
        callbacks=[checkpoint_callback, lr_monitor], 
        logger=logger,
        train_config=config,
    )
    save_config(config, savedir, prefix="train")
    trainer.fit(
        model=model, train_dataloaders=train_dataloader, build_dataloader=build_dataloader, val_dataloaders=benchmark_dataloader
    )


def main():
    parser = create_argparse()
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
