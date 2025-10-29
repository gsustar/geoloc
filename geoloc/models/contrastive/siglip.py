import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning
import itertools

import transformers

from lightning.pytorch.utilities import grad_norm
from .image_encoder import ImageEncoder

class SigLIPLoss(nn.Module):
	def __init__(self):
		super().__init__()

		temperature_init = 10.0
		bias_init = -10.0
		self.temp = nn.Parameter(torch.log(torch.tensor(temperature_init, dtype=torch.float32)))
		self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))

	def forward(self, embeddings):
		B, n, d = embeddings.shape
		loss = 0.0
		for i, j in itertools.combinations(range(n), 2):
			logits = torch.matmul(embeddings[:, i], embeddings[:, j].T) * torch.exp(self.temp) + self.bias
			eye = torch.eye(B).to(logits.device)
			labels = 2 * eye - torch.ones_like(logits)
			loglik = F.logsigmoid(labels * logits)
			nll = -torch.sum(loglik, axis=-1)
			loss += torch.mean(nll)
		return loss

class SigLIPModel(lightning.LightningModule):
	def __init__(self):
		super().__init__()
		self.image_encoder = ImageEncoder()
		self.loss_fn = SigLIPLoss()

	def forward(self, x):
		x = self.image_encoder(x)
		x = F.normalize(x, dim=-1)
		return x

	def training_step(self, batch, batch_idx):
		# batch: [B, num_same_place, C, H, W]
		images = batch["image"]
		B, n, C, H, W = images.shape
		embeddings = []
		for i in range(B):
			embed = self.image_encoder(images[i])
			embeddings.append(embed)
		embeddings = torch.stack(embeddings, dim=0)
		embeddings = F.normalize(embeddings, dim=-1)
		loss = self.loss_fn(embeddings)
		self.log("train_loss", loss, prog_bar=True, on_step=True)
		return loss
	
	def on_before_optimizer_step(self, optimizer):
		norms = grad_norm(self.image_encoder, norm_type=2)
		self.log_dict(norms)

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-7)
		num_warmup_steps = 6500
		num_training_steps = self.trainer.estimated_stepping_batches
		scheduler = transformers.get_cosine_schedule_with_warmup(
			optimizer,
			num_warmup_steps=num_warmup_steps,
			num_training_steps=num_training_steps
		)
		return {
			"optimizer": optimizer,
			"lr_scheduler": {
				"scheduler": scheduler,
				"interval": "step",
				"frequency": 1
			}
		}