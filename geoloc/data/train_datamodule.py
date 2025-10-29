import torch
import lightning as L
from ..config_parser import class_from_config

class TrainingDataModule(L.LightningDataModule):
	def __init__(self, dataset_config, dataloader_config):
		super().__init__()
		self.dataset_config = dataset_config
		self.dataloader_config = dataloader_config
		self.save_hyperparameters()

	def setup(self, stage):
		self.dataset = class_from_config(self.dataset_config)

	def train_dataloader(self):
		return torch.utils.data.DataLoader(self.dataset, **vars(self.dataloader_config))