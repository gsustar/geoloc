import argparse
import datetime
import lightning as L

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from geoloc.config_parser import load_config, namespace_to_dict
from geoloc.data.train_datamodule import TrainingDataModule
from geoloc.relatedwork.salad import SALADModel


def create_argparse():
    parser = argparse.ArgumentParser(description="SALAD Trainer for VPAir")
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
    model = SALADModel(
        backbone_config=config.model.backbone,
        aggregator_config=config.model.aggregator,
        loss_config=config.model.loss,
        miner_config=config.model.miner,
        optimizer_config=config.model.optimizer,
        scheduler_config=config.model.scheduler,
    )
    if hasattr(config.model, "_pretrained_ckpt_path"):
        model.load_pretrained(config.model._pretrained_ckpt_path)

    dataset_name = f"{config.datamodule.dataset.class_path.split('.')[-2]}"
    run_name = (
        "SALAD_"
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
        group="SALAD",
        save_dir="/home/grega/geoloc/logs/",
        name=run_name,
        log_model=False,
        config=namespace_to_dict(config),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/home/grega/geoloc/logs/checkpoints/SALAD/{dataset_name}/",
        filename=run_name + "-{epoch:02d}-{step:06d}",
    )
    trainer = L.Trainer(
        **vars(config.trainer), callbacks=[checkpoint_callback], logger=logger
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
