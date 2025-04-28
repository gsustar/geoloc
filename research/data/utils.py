import os
import torch
import torchvision
from natsort import natsorted
from typing import List
import torchvision.transforms.functional as F

def is_image(path_or_filename: str) -> bool:
	""" Check if path is an image file """
	return path_or_filename.lower().endswith((".jpg", ".png", ".jpeg"))


def get_sorted_imgpaths(directory: str) -> List[str]:
	""" Get sorted list of absolute image paths in directory """
	return natsorted([
		os.path.join(directory, img)
		for img in os.listdir(directory)
		if is_image(os.path.join(directory, img))
	])


def load_image(path: str) -> torch.Tensor:
	""" Load an image as a float32 torch.Tensor """
	return torchvision.io.read_image(path).to(torch.float32)


def pad_to_size(img: torch.Tensor, target_size: int) -> torch.Tensor:
	""" Zero-pad image to target size """
	_, h, w = img.shape
	pad_h = max(0, target_size - h)
	pad_w = max(0, target_size - w)
	ptile = F.pad(img, (0, 0, pad_w, pad_h), fill=0)
	return ptile
