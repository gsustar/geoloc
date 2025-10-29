import os
import yaml
import torch
import numpy as np
import pandas as pd

from ..utils import load_image, get_sorted_imgpaths
from .base import ResizeCenterCropMixin


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

def _load_gt_pos(root):
	return np.load(
		os.path.join(root, "vpair_gt.npy"), 
		allow_pickle=True
	)

class VPAirReferenceImages(torch.utils.data.Dataset, ResizeCenterCropMixin):

	def __init__(
		self,
		root: str, 
		include_distractors: bool = True,
		resize=None,
		center_crop=None
	):
		super().__init__(resize=resize, center_crop=center_crop)
		self.root = root
		self.crs = "EPSG:4326"
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
		img = self.resize_centercrop(img)
		img = img / 255.0
		filename = os.path.basename(self.reference_images[index])
		# Distractors do not have pose information
		lon = 0.0 if is_distractor else self.poses["lon"].iloc[index]
		lat = 0.0 if is_distractor else self.poses["lat"].iloc[index]
		alt = 0.0 if is_distractor else self.poses["altitude"].iloc[index]
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

class VPAirQueryImages(torch.utils.data.Dataset, ResizeCenterCropMixin):

	def __init__(self, root: str, soft_positive_offset: int = 3, resize=None, center_crop=None):
		super().__init__(resize=resize, center_crop=center_crop)
		self.root = root
		self.crs = "EPSG:4326"
		self.queries = _load_queries(root)
		self.poses = _load_poses(root)
		# self.gt_pos = _load_gt_pos(root)
		self.soft_positive_offset = soft_positive_offset

	def __len__(self):
		return len(self.queries)
	
	def __getitem__(self, index):
		img = load_image(self.queries[index])
		img = self.resize_centercrop(img)
		img = img / 255.0
		filename = os.path.basename(self.queries[index])
		lon = self.poses["lon"].iloc[index]
		lat = self.poses["lat"].iloc[index]
		alt = self.poses["altitude"].iloc[index]
		gt_pos = np.unique(np.clip(
			list(range(
				# (index + 1) - self.soft_positive_offset, 
				# (index + 1) + self.soft_positive_offset + 1
				index - self.soft_positive_offset, 
				index + self.soft_positive_offset + 1
			)), 0, len(self.queries)
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


VPAIR_TRAIN_TEST_SPLIT = 0.7
class VPAirTrainDataset(torch.utils.data.Dataset, ResizeCenterCropMixin):

	def __init__(
 		self, root: str, resize=None, center_crop=None
	):
		super().__init__(resize=resize, center_crop=center_crop)
		self.root = root
		self.crs = "EPSG:4326"

		self.queries = _load_queries(root)
		self.reference_views = _load_reference_views(root)

		train_cut = int(np.floor(len(self.queries) * VPAIR_TRAIN_TEST_SPLIT))
		self.queries = self.queries[:train_cut]
		self.reference_views = self.reference_views[:train_cut]
		self.poses = _load_poses(root).iloc[:train_cut].reset_index(drop=True)
		# self.poses = self.poses.iloc[:train_cut].reset_index(drop=True)

	def __len__(self):
		assert len(self.queries) == len(self.reference_views)
		return len(self.queries)

	def __getitem__(self, index):
		query = load_image(self.queries[index])
		query = self.resize_centercrop(query)
		query = query / 255.0
		
		ref = load_image(self.reference_views[index])
		ref = self.resize_centercrop(ref)
		ref = ref / 255.0

		lon = self.poses["lon"].iloc[index]
		lat = self.poses["lat"].iloc[index]
		alt = self.poses["altitude"].iloc[index]

		imgs = torch.stack([query, ref], dim=0)
		return dict(
			images=imgs, # [N, C, H, W], where N is the number of same-place images (2 here)
			index=index,
			lon=lon,
			lat=lat,
			alt=alt
		)


class VPAirTestDatasetQueryImages(VPAirQueryImages):

	def __init__(
 		self, 
		root: str, 
		soft_positive_offset: int = 3, 
		resize=None, 
		center_crop=None
	):
		super().__init__(root=root, soft_positive_offset=soft_positive_offset, resize=resize, center_crop=center_crop)
		train_cut = int(np.floor(len(self.queries) * VPAIR_TRAIN_TEST_SPLIT))
		self.queries = self.queries[train_cut:]
		self.poses = self.poses.iloc[train_cut:].reset_index(drop=True)


class VPAirTestDatasetReferenceImages(VPAirReferenceImages):
	
	def __init__(
 		self, 
		root: str, 
		include_distractors: bool = True,
		resize=None, 
		center_crop=None
	):
		super().__init__(root=root, include_distractors=include_distractors, resize=resize, center_crop=center_crop)
		train_cut = int(np.floor(self.num_reference_views * VPAIR_TRAIN_TEST_SPLIT))
		self.num_reference_views = self.num_reference_views - train_cut
		self.reference_images = self.reference_images[train_cut:]
		self.poses = self.poses.iloc[train_cut:].reset_index(drop=True)