import os
import argparse
import datetime
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from geoloc.config_parser import load_config, save_config, namespace_to_dict, class_from_config
from geoloc.data.train_datamodule import TrainingDataModule


def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    return parser

def main():
    parser = create_argparse()
    args = parser.parse_args()
    config = load_config(args.config)

    datamodule = TrainingDataModule(
        dataset_config=config.datamodule.dataset,
        dataloader_config=config.datamodule.dataloader,
    )
    model = class_from_config(config.model)
    model_cls = config.model.class_path.split(".")[-1].lower()

    if hasattr(config.model, "_pretrained_ckpt_path"):
        model.load_pretrained(config.model._pretrained_ckpt_path)

    dataset_name = f"{config.datamodule.dataset.class_path.split('.')[-2]}"
    run_name = (
        f"{model_cls}_"
        f"{dataset_name}_"
        f"{config.model.backbone.class_path.split('.')[-1]}_"
        f"N-align={config.datamodule.dataset.init_args.north_align}_"
        f"lr={config.model.optimizer.init_args.lr:.0e}_"
        f"bs={config.datamodule.dataloader.batch_size}_"
        f"ep={config.trainer.max_epochs}"
        f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger = WandbLogger(
        project="geoloc",
        group=model_cls,
        save_dir="/home/grega/geoloc/logs/",
        name=run_name,
        log_model=False,
        config=namespace_to_dict(config),
    )
    savedir = f"/home/grega/geoloc/logs/checkpoints/{model_cls}/{dataset_name}/"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/home/grega/geoloc/logs/checkpoints/{model_cls}/{dataset_name}/",
        filename=run_name + "-{epoch:02d}-{step:06d}",
    )
    trainer = L.Trainer(
        **vars(config.trainer), callbacks=[checkpoint_callback], logger=logger
    )
    trainer.fit(model=model, datamodule=datamodule)
    save_config(config, savedir, prefix="fit")

if __name__ == "__main__":
    main()
