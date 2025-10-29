import os
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm
import einops

from ..utils import DEBUG

PATH_TO_PREFITTED_CENTERS = "/storage/datasets/AerialLoc/Drone2Sat/ckpts/anyloc_aerial_c_centers/c_centers.pt"

class AnyLoc(nn.Module):
	def __init__(
		self,
		backbone,
		vlad,
		pca=None,
	):
		super().__init__()
		self.backbone = backbone
		self.vlad = vlad
		self.pca = pca

	# def _load_prefitted_vlad(self):
	# 	vlad_pre = torch.hub.load("AnyLoc/DINO", "get_vlad_model", 
	# 				backbone="DINOv2", domain="aerial").vlad
	# 	assert (self.vlad.mode == "cosine" and
	# 	  		self.vlad.vlad_mode == "hard" and
	# 			self.vlad.desc_dim == vlad_pre.desc_dim and
	# 			self.vlad.num_clusters == vlad_pre.num_clusters), (
	# 	"Pre-fitted VLAD model parameters do not match the configuration")
	# 	vlad_pre_c_centers = vlad_pre.c_centers.cpu().clone()
	# 	self.vlad.fit(None, c_centers=vlad_pre_c_centers)
	# 	del vlad_pre
	# 	return self.vlad
	
	def _load_prefitted_vlad(self):
		assert (self.vlad.mode == "cosine" and
		  		self.vlad.vlad_mode == "hard" and
				self.vlad.desc_dim == 1536 and
				self.vlad.num_clusters == 32), (
		"Pre-fitted VLAD model parameters do not match the configuration")
		vlad_pre_c_centers = torch.load(PATH_TO_PREFITTED_CENTERS)
		self.vlad.fit(None, c_centers=vlad_pre_c_centers)
		return self.vlad

	def fit_and_generate(self, ref_image_dataloader, savedir,
					  vlad_fitstep=1, use_prefitted_vlad=False,
					  device="cpu"):
		print("Extracting features...")
		features = []
		for i, batch in enumerate(tqdm(ref_image_dataloader)):
			if DEBUG > 1 and i > 20:
				break
			if i % vlad_fitstep != 0:
				continue
			image = batch["image"].to(device)
			x = self.backbone(image)
			B, C, _, _ = x.shape
			x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
			features.append(x.detach())
			if self.pca is None and use_prefitted_vlad:
				break # Early exit in case of pre-fitted VLAD and no PCA, only need features for asserting vector dimension
		features = torch.vstack(features)

		if use_prefitted_vlad:
			self.vlad = self._load_prefitted_vlad()
			features = self.vlad.generate_multi(features)
			self.vlad.save() # Manually save `c_centers` (usually this happens during `fit`)
		else:
			print("Fitting VLAD features...")
			assert self.vlad.cache_dir is not None
			assert self.vlad.cache_dir == savedir
			features = self.vlad.fit_and_generate(features)

		# TODO: test if PCA works
		if self.pca is not None:
			print("Fitting PCA...")
			features = self.pca.fit_transform(features.cpu())
			self.save_pca(savedir)
			print(f"PCA components: {self.pca.components_.shape}")
		return features

	def load_vlad(self, loaddir):
		assert self.vlad.cache_dir is not None
		assert self.vlad.cache_dir == loaddir
		assert os.path.exists(os.path.join(loaddir, "c_centers.pt"))
		self.vlad.fit(None)

	def load_pca(self, loaddir):
		pca_loadpath = os.path.join(loaddir, "pca.pkl")
		if os.path.exists(pca_loadpath):
			print("Loading PCA from file...")
			with open(pca_loadpath, "rb") as f:
				self.pca = pickle.load(f)

	def save_pca(self, savedir):
		if self.pca is not None:
			with open(os.path.join(savedir, "pca.pkl"), "wb") as f:
				pickle.dump(self.pca, f)

	def forward(self, x, return_residuals=False):
		x = self.backbone(x)
		x = einops.rearrange(x, "b c h w -> b (h w) c")
		if return_residuals:
			qry_residuals = self.vlad.generate_multi_res_vec(x)
		x = self.vlad.generate_multi(x)
		x = torch.from_numpy(self.pca.transform(x)).float() if self.pca is not None else x

		if return_residuals:
			return x, qry_residuals
		return x