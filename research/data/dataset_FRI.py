import os
import torch
import json
import torchvision.transforms.functional as F
from .utils import load_image, get_sorted_imgpaths
from .combine_tiles import construct_image_from_metadata


class FRIReferenceImages(torch.utils.data.Dataset):
	"""
	Reference images are images that cover a rectangular area around FRI.
	"""
	
	def __init__(self, root, resize=None):
		super().__init__()
		self.root = root
		self.resize = resize

		# Open metadata file
		metadata_path = os.path.join(root, "metadata.json")
		assert os.path.exists(metadata_path), (
			f"Metadata file does not exist: {os.path.join(root, 'metadata.json')}"
		)
		with open(metadata_path, 'r') as f:
			self.metadata = json.load(f)
		self.reference_images = list(self.metadata["images"].keys())

	def __len__(self):
		return len(self.reference_images)

	def __getitem__(self, index):
		try:
			img = load_image(self.reference_images[index])
		except RuntimeError as e:
			# construct image on-the-fly if the image is not found but metadata is available
			img = construct_image_from_metadata(
				self.metadata,
				self.reference_images[index],
				self.root,
			)
			img = F.pil_to_tensor(img)

		img = img / 255.0
		if self.resize is not None:
			img = F.resize(img, self.resize)
		filename = os.path.basename(self.reference_images[index])
		img_metadata = self.metadata["images"][self.reference_images[index]]
		lon = img_metadata["center"]["lon"]
		lat = img_metadata["center"]["lat"]
		return dict(
			image=img, 
			filename=filename,
			index=index,
			lon=lon,
			lat=lat,
		)


class FRIQueryImages(torch.utils.data.Dataset):
	"""
	Query image are image obtained from am UAV trajectory over a portion of the FRI area.
	"""

	def __init__(self, root, resize=None):
		super().__init__()
		self.root = root
		self.query_images = get_sorted_imgpaths(root)  # Load query images from the root directory
		self.resize = resize

		# Open metadata file
		metadata_path = os.path.join(root, "metadata.json")
		with open(metadata_path, 'r') as f:
			self.metadata = json.load(f)


	def __len__(self):
		return len(self.query_images)


	def __getitem__(self, index):
		img = load_image(self.query_images[index])
		img = img / 255.0
		if self.resize is not None:
			img = F.resize(img, self.resize)
		filename = os.path.basename(self.query_images[index])
		img_metadata = self.metadata[self.query_images[index]]
		lon = img_metadata["lon"]
		lat = img_metadata["lat"]
		return dict(
			image=img, 
			filename=filename,
			index=index,
			lon=lon,
			lat=lat,
		)


class GESQueryImages(torch.utils.data.Dataset):
	"""
	Query image are image obtained from am UAV trajectory over a portion of the Ljubljana area.
	"""

	def __init__(self, root, resize=None):
		super().__init__()
		self.root = root

		self.trajectory_name = os.path.basename(os.path.dirname(root))
		self.footage_dir = os.path.join(root, "footage")

		self.query_images = get_sorted_imgpaths(self.footage_dir)  # Load query images from the root directory
		self.resize = resize

		self.metadata_path = os.path.join(root, f"{self.trajectory_name}.json")
		with open(self.metadata_path, 'r') as f:
			self.metadata = json.load(f)


	def __len__(self):
		return len(self.query_images)

	def __getitem__(self, index):
		img = load_image(self.query_images[index])
		img = img / 255.0
		if self.resize is not None:
			img = F.resize(img, self.resize)
		filename = os.path.basename(self.query_images[index])
		lon = self.metadata["cameraFrames"][index]["coordinate"]["longitude"]
		lat = self.metadata["cameraFrames"][index]["coordinate"]["latitude"]
		return dict(
			image=img, 
			filename=filename,
			index=index,
			lon=lon,
			lat=lat,
		)


if __name__ == "__main__":
	ref_database_root = "/storage/datasets/AerialLoc/Drone2Sat/FRISat/arcgis/combined/1024x1024/2025/zoom19/"
	qry_database_root = "/storage/datasets/AerialLoc/Drone2Sat/FRISat/trajectories/FRI-110m-sunny/queries/"

	ref_image_dataset = FRIReferenceImages(root=ref_database_root)
	print(ref_image_dataset[0])
	qry_image_dataset = FRIQueryImages(root=qry_database_root)
	print(qry_image_dataset[0])
	
	print("all okay!")