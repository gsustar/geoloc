import os
import yaml
import torch
import numpy as np
import pandas as pd

from .utils import load_image, get_sorted_imgpaths


def _load_poses(root):
	return pd.read_csv(os.path.join(root, "poses.csv"))

def _load_camera_calibration(root):
	with open(os.path.join(root, "camera_calibration.yaml"), "r") as f:
		camera_calibration = yaml.safe_load(f)
	return camera_calibration

def _load_distractors(root):
	return get_sorted_imgpaths(
		os.path.join(root, "distractors")
	)

def _load_reference_views(root):
	return get_sorted_imgpaths(
		os.path.join(root, "reference_views")
	)

def _load_queries(root):
	return get_sorted_imgpaths(
		os.path.join(root, "queries")
	)


class VPAirReferenceImages(torch.utils.data.Dataset):

	def __init__(
		self,
		root: str, 
		include_distractors: bool = True
	):
		self.root = root
		self.include_distractors = include_distractors

		self.distractors = _load_distractors(root)
		self.reference_views = _load_reference_views(root)
		self.num_reference_views = len(self.reference_views)
		self.num_distractors = len(self.distractors)
		self.poses = _load_poses(root)

		# Optionally include distractors in reference images
		self.reference_images = self.reference_views.copy()
		if self.include_distractors:
			self.reference_images.extend(self.distractors)

		del self.distractors
		del self.reference_views

	def __len__(self):
		return len(self.reference_images)

	def __getitem__(self, index):
		is_distractor = index >= self.num_reference_views
		img = load_image(self.reference_images[index])
		img = img / 255.0
		filename = os.path.basename(self.reference_images[index])
		# Distractors do not have pose information
		lon = None if is_distractor else self.poses["lon"].iloc[index]
		lat = None if is_distractor else self.poses["lat"].iloc[index]
		alt = None if is_distractor else self.poses["altitude"].iloc[index]
		return dict(
			image=img, 
			filename=filename,
			index=index,
			lon=lon,
			lat=lat,
			alt=alt,
			is_distractor=is_distractor,
			gt_pos=-1,
		)


class VPAirQueryImages(torch.utils.data.Dataset):
	
	def __init__(self, root: str, soft_positive_offset: int = 0):
		self.root = root
		self.queries = _load_queries(root)
		self.poses = _load_poses(root)
		self.soft_positive_offset = soft_positive_offset

	def __len__(self):
		return len(self.queries)
	
	def __getitem__(self, index):
		img = load_image(self.queries[index])
		img = img / 255.0
		filename = os.path.basename(self.queries[index])
		lon = self.poses["lon"].iloc[index]
		lat = self.poses["lat"].iloc[index]
		alt = self.poses["altitude"].iloc[index]
		gt_pos = np.unique(np.clip(
			list(range(
				index-self.soft_positive_offset, 
				index+self.soft_positive_offset+1
			)), 0, len(self.queries)-1
		))
		return dict(
			image=img, 
			filename=filename,
			index=index,
			lon=lon,
			lat=lat,
			alt=alt,
			gt_pos=gt_pos,
		)
	

if __name__ == "__main__":
	database_root = "/storage/datasets/AerialLoc/Drone2Sat/VPAir"
	vpair_include_distractors = False
	soft_positive_offset = 3

	ref_image_dataset = VPAirReferenceImages(root=database_root, include_distractors=vpair_include_distractors)
	qry_image_dataset = VPAirQueryImages(root=database_root, soft_positive_offset=soft_positive_offset)
	print("all okay!")