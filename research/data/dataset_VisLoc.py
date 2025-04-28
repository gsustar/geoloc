import os
import math
import torch
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rasterio.windows import Window
from rasterio.transform import xy
import torchvision.transforms.functional as F

from .utils import pad_to_size, get_sorted_imgpaths, load_image


class VisLocReferenceImages(torch.utils.data.Dataset):
	def __init__(self, root, flight_idx, tile_size=256, stride=None, resize=None):
		super().__init__()
		self.root = root
		self.flight_idx = f"{int(flight_idx):02d}"
		self.tif_path = os.path.join(self.root, self.flight_idx, f"satellite{self.flight_idx}.tif")
		self.tile_size = tile_size
		self.stride = stride if stride else tile_size
		self.resize = resize

		self.sat_coords = pd.read_csv(
			os.path.join(self.root, "satellite_ coordinates_range.csv")
		).iloc[int(self.flight_idx) - 1]

		with rasterio.open(self.tif_path) as src:
			self.tif_width = src.width
			self.tif_height = src.height
			self.count = src.count  # Number of bands
			self.ul = [src.bounds.left, src.bounds.top]  # Upper left corner
			self.br = [src.bounds.right, src.bounds.bottom]  # Bottom right corner
			self.transform = src.transform

		# Precompute all valid tile windows
		self.tiles = []
		for x in range(0, self.tif_width, self.stride):
			for y in range(0, self.tif_height, self.stride):
				w = min(self.tile_size, self.tif_width - x)
				h = min(self.tile_size, self.tif_height - y)
				self.tiles.append((x, y, w, h))

		self.num_tiles_width = math.ceil(self.tif_width / self.stride)
		self.num_tiles_height = math.ceil(self.tif_height / self.stride)

	def __len__(self):
		return len(self.tiles)

	def __getitem__(self, index):
		x, y, w, h = self.tiles[index]
		window = Window(x, y, w, h)

		with rasterio.open(self.tif_path) as src:
			tile = src.read(window=window)  # (bands, h, w)
			# transform = src.window_transform(window)

		# Normalize to [0, 1]
		tile = tile.astype(np.float32) / 255.0
		tile = torch.from_numpy(tile)  # shape: (C, H, W)

		# Zero-pad if edgetile
		tile = pad_to_size(tile, self.tile_size)

		assert tile.shape[1] == tile.shape[2] == self.tile_size, (
			f"Tile shape mismatch: {tile.shape} != ({self.tile_size}, {self.tile_size})"
		)
		# Resize if requested
		if self.resize is not None:
			tile = F.resize(tile, self.resize)

		# Get center coordinates (lat/lon or projection units)
		center_col = x + w // 2
		center_row = y + h // 2
		center_x, center_y = xy(self.transform, center_row, center_col)

		return dict(
			image=tile,
			index=index,
			filename="",
			lon=center_x,
			lat=center_y,
		)


class VisLocQueryImages(torch.utils.data.Dataset):
	
	def __init__(self, root, flight_idx, resize=None):
		super().__init__()
		self.root = root
		self.flight_idx = f"{int(flight_idx):02d}"
		self.resize = resize
		self.query_images = get_sorted_imgpaths(os.path.join(self.root, self.flight_idx, "drone"))
		self.query_metadata = pd.read_csv(
			os.path.join(self.root, self.flight_idx, f"{self.flight_idx}.csv")
		)

	def __len__(self):
		return len(self.query_images)
	
	def __getitem__(self, index):
		img = load_image(self.query_images[index])
		img = img / 255.0
		if self.resize is not None:
			img = F.resize(img, self.resize)
		filename = os.path.basename(self.query_images[index])
		img_metadata = self.query_metadata.iloc[index]
		lon = img_metadata["lon"] # lon and lat of center of the image
		lat = img_metadata["lat"]
		return dict(
			image=img,
			index=index,
			filename=filename,
			lon=lon,
			lat=lat,
		)

if __name__ == "__main__":
	root = "/storage/datasets/AerialLoc/UAV_VisLoc"
	ref_image_dataset = VisLocReferenceImages(root=root, flight_idx=1, tile_size=256, stride=256)
	tile = ref_image_dataset[10]["image"]
	plt.imshow(tile.permute(1, 2, 0))  # Transpose to (H, W, C) for plotting
	plt.savefig("tile.png")
	query_image_dataset = VisLocQueryImages(root=root, flight_idx=1, resize=(256, 256))
	query = query_image_dataset[0]["image"]
	plt.imshow(query.permute(1, 2, 0))  # Transpose to (H, W, C) for plotting
	plt.savefig("query.png")

	print("all okay!")