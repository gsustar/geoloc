import argparse
import lightning as L

from geoloc.config_parser import load_config
from geoloc.data.train_datamodule import TrainingDataModule
from geoloc.relatedwork.mixvpr import MixVPRModel


def create_argparse():
	parser = argparse.ArgumentParser(description="MixVPR Trainer for VPAir")
	parser.add_argument("--config", type=str, help="Path to the configuration file")
	return parser


def main():
	parser = create_argparse()
	args = parser.parse_args()
	config = load_config(args.config)

	datamodule = TrainingDataModule(
		dataset_config=config.datamodule.dataset,
		dataloader_config=config.datamodule.dataloader
	)
	model = MixVPRModel(
		backbone_config=config.model.backbone,
		aggregator_config=config.model.aggregator,
		loss_config=config.model.loss,
		miner_config=config.model.miner,
		optimizer_config=config.model.optimizer,
		scheduler_config=config.model.scheduler,
	)

	trainer = L.Trainer(**vars(config.trainer), callbacks=[])
	trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
	main()