import sys
import os
sys.path.append("/home/grega/")
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #* Set appropriate GPU
os.environ["CPL_LOG"] = "ERROR"

import argparse
import torch
import lightning
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from geoloc.experiments.contrastive.siglip import SigLIPModel
from geoloc.experiments.contrastive.clip import CLIPModel
from geoloc.data.datasets.GURS import TrainGURSDataset, TifAwareShuffleSampler

class GURSDataModule(lightning.LightningDataModule):
	def __init__(
		self, 
		rootdir, 
		tile_size=250, 
		border="Ljubljana", 
		exclude_slo_border_tifs=True, 
		num_same_place=3,
		batch_size=16,
		num_workers=0
	):
		super().__init__()
		self.rootdir = rootdir
		self.tile_size = tile_size
		self.border = border
		self.exclude_slo_border_tifs = exclude_slo_border_tifs
		self.num_same_place = num_same_place

		self.batch_size = batch_size
		self.num_workers = num_workers

	def setup(self, stage=None):
		self.train_dataset = TrainGURSDataset(
			rootdir=self.rootdir,
			tile_size=self.tile_size,
			border=self.border,
			exclude_slo_border_tifs=self.exclude_slo_border_tifs,
			num_same_place=self.num_same_place,
		)

	def train_dataloader(self):
		# [B, num_same_place, C, H, W]
		# Make sure to handle the batch size correctly when passing to the model
		sampler = TifAwareShuffleSampler(
			self.train_dataset.num_tifs,
			self.train_dataset.num_windows_per_tif,
			N=3
		)
		return torch.utils.data.DataLoader(
			self.train_dataset, 
			batch_size=self.batch_size, 
			shuffle=False,
			num_workers=self.num_workers,
			sampler=sampler
		)

	def teardown(self, stage):
		self.train_dataset.teardown()
		return super().teardown(stage)

def parse_args():
	parser = argparse.ArgumentParser()
	# DataModule parameters
	parser.add_argument("--rootdir", type=str, default="/storage/private/MORS/gurs/")
	parser.add_argument("--tile_size", type=int, default=250)
	parser.add_argument("--border", type=str, default="Ljubljana")
	parser.add_argument("--exclude_slo_border_tifs", action="store_true")
	parser.add_argument("--num_same_place", type=int, default=3)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--model_type", type=str, default="siglip", choices=["siglip", "clip"])

	# Trainer parameters
	parser.add_argument("--max_steps", type=int, default=100000)
	parser.add_argument("--accelerator", type=str, default="gpu")
	parser.add_argument("--devices", type=int, default=1)
	parser.add_argument("--precision", type=str, default="16-mixed")
	parser.add_argument("--default_root_dir", type=str, default="/home/grega/geoloc/logs/")
	parser.add_argument("--accumulate_grad_batches", type=int, default=1)
	return parser.parse_args()

def main():
	args = parse_args()

	if args.model_type == "siglip":
		model = SigLIPModel()
	elif args.model_type == "clip":
		model = CLIPModel()
	else:
		raise ValueError(f"Unsupported model type: {args.model_type}")

	data_module = GURSDataModule(
		rootdir=args.rootdir,
		tile_size=args.tile_size,
		border=args.border,
		exclude_slo_border_tifs=args.exclude_slo_border_tifs,
		num_same_place=args.num_same_place,
		batch_size=args.batch_size,
		num_workers=args.num_workers
	)

	lr_monitor = LearningRateMonitor(logging_interval="step")
	wandb_logger = WandbLogger(
		project="geoloc",
		name=f"contrastive_{args.model_type}",
		save_dir=args.default_root_dir,
	)
	wandb_logger.watch(model, log="all")
	trainer = lightning.Trainer(
		max_steps=args.max_steps,
		accelerator=args.accelerator,
		devices=args.devices,
		precision=args.precision,
		default_root_dir=args.default_root_dir,
		accumulate_grad_batches=args.accumulate_grad_batches,
		callbacks=[lr_monitor],
		logger=wandb_logger,
	)
	
	trainer.fit(
		model=model,
		datamodule=data_module
	)

if __name__ == "__main__":
	main()