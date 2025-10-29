import math
import torch.nn as nn

def freeze(model: nn.Module):
	"""Freeze the model parameters."""
	for param in model.parameters():
		param.requires_grad = False
	return model