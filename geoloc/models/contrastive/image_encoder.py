import torch
import torch.nn as nn

from ..backbones import HFaceDINOv2Backbone, HFaceDINOv3Backbone


class AdapterHead(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.aggregate = lambda x: torch.mean(x, dim=1) # Aggregate over the sequence dimension
		self.adapter = nn.Sequential(
			nn.Linear(input_dim, output_dim),
			nn.ReLU(),
			nn.Linear(output_dim, output_dim)
		)

	def forward(self, x):
		x = self.aggregate(x)
		return self.adapter(x)

class ImageEncoder(nn.Module):
	def __init__(
		self, 
		backbone_type="dinov3",
		output_dim=1024,
	):
		super().__init__()
		assert backbone_type in ["dinov2", "dinov3"], "Invalid backbone type: {backbone_type}"
		self.backbone_type = backbone_type
		self.output_dim = output_dim

		if self.backbone_type == "dinov2":
			self.backbone = HFaceDINOv2Backbone()
		elif self.backbone_type == "dinov3":
			self.backbone = HFaceDINOv3Backbone()

		self.adapter = AdapterHead(
			input_dim=self.backbone.output_dim,
			output_dim=output_dim
		)

	def forward(self, x):
		x = self.backbone(x)
		x = self.adapter(x)
		return x