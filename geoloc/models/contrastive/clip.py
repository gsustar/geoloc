import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning
import itertools

from lightning.pytorch.utilities import grad_norm

from .image_encoder import ImageEncoder

class CLIPLoss(nn.Module):
	def __init__(self):
		super().__init__()
		temperature_init = 0.07
		self.temp = nn.Parameter(torch.log(torch.tensor(temperature_init, dtype=torch.float32)))

	def forward(self, embeddings):
		B, n, d = embeddings.shape
		loss = 0.0
		for i, j in itertools.combinations(range(n), 2):
			self.temp
			logits = torch.matmul(embeddings[:, i], embeddings[:, j].T) * torch.exp(self.temp)
			labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
			loss += F.cross_entropy(logits, labels)
		return loss

class CLIPModel(lightning.LightningModule):
	def __init__(self):
		super().__init__()
		self.image_encoder = ImageEncoder()
		self.loss_fn = CLIPLoss()

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

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
		return optimizer
	
	def on_before_optimizer_step(self, optimizer):
		norms = grad_norm(self.image_encoder, norm_type=2)
		self.log_dict(norms)