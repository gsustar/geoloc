"""
This file contains the code and results of the experiments for the AnyLoc method on the FRI dataset.
"""
import os
import cv2
import sys
research_dir = f"{os.path.dirname(os.path.realpath(__file__))}/../"
sys.path.insert(1, research_dir)

import torchvision
import torch
from typing import Union
import numpy as np
from PIL import Image
from tqdm import tqdm
import tracemalloc
import time
import geopy.distance
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

from AnyLoc.utilities import DinoV2ExtractFeatures
from research.data.vdb import VectorDatabase, load_database
from research.data.dataset_FRI import FRIReferenceImages, FRIQueryImages, GESQueryImages
from research.aggregators.vlad import VLAD

def color_map_color(value, cmap_name='jet', vmin=0, vmax=1):
	norm = plt.Normalize(vmin, vmax)
	cmap = plt.get_cmap(cmap_name)
	rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
	return rgb

#*############## Global arguments ##############*#

DEBUG = int(os.environ.get("DEBUG", 0))
PLOT = int(os.environ.get("PLOT", 0))
PLOT_HEATMAP = int(os.environ.get("PLOT_HEATMAP", 0))

ROTTRAJ_EXP = int(os.environ.get("ROTTRAJ_EXP", 0))
ROTREF_EXP = int(os.environ.get("ROTREF_EXP", 0))

VISUALIZE_VLAD = int(os.environ.get("VISUALIZE_VLAD", 0))

#*############## Preprocessor ##############*#

class AnyLocPreprocessor:

	def __call__(self, image: Union[torch.Tensor, Image.Image, np.ndarray], device: torch.device = None, **kwargs) -> torch.Tensor:
		assert image.dim() in [3, 4], "Input image must have 3 or 4 dimensions"
		if isinstance(image, (Image.Image, np.ndarray)):
			image = torchvision.transforms.functional.to_tensor(image)
		image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
		h, w = image.shape[-2:]
		h_new, w_new = (h // 14) * 14, (w // 14) * 14
		image = torchvision.transforms.functional.center_crop(image, (h_new, w_new))
		if image.dim() == 3:
			image = image.unsqueeze(0)
		return image.to(device)


#*############## Utility functions ##############*#

def bytes_to_gb(bytes: int):
	return bytes / (1024**3)

#*############## main ##############*#

@torch.no_grad()
def main():

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	long_trajectory = True

	# DINOv2 Arguments
	dinov2_ckpt = "dinov2_vitg14"
	dinov2_desc_layer = 31
	dinov2_desc_facet = "value"

	# VLAD Arguments
	vlad_num_clusters = 32
	vlad_mode = "hard"
	cache_dir = None
	vlad_minibatch = 1024
	vlad_kmeans_type = "minibatch"
	vlad_fit_step = 5
	vlad_anyloc_prefitted = True

	dataloader_batchsize = 16

	save_faiss_index = True
	load_faiss_index = False
	load_faiss_index_dir = "/storage/datasets/AerialLoc/Drone2Sat/LjubljanaSat/results/anyloc/database_norot" # nezarotirana baza 256x256
	if ROTREF_EXP:
		load_faiss_index_dir = "/storage/datasets/AerialLoc/Drone2Sat/LjubljanaSat/results/anyloc/database_rot" # 4x zarotirana baza 256x56

	# base_dir = "/storage/datasets/AerialLoc/Drone2Sat/FRISat/"
	base_dir = "/storage/datasets/AerialLoc/Drone2Sat/LjubljanaSat/"

	size = 1024
	ref_database_root = os.path.join(base_dir, f"arcgis/combined/{size}x{size}/2025/zoom19/")

	resize_size = (512, 512)
	ref_image_dataset = FRIReferenceImages(root=ref_database_root, resize=resize_size)
	ref_image_dataloader = DataLoader(ref_image_dataset, batch_size=dataloader_batchsize, shuffle=False)

	qry_database_root = "/storage/datasets/AerialLoc/Drone2Sat/FRISat/trajectories/FRI-110m-sunny/queries/"
	qry_image_dataset = FRIQueryImages(root=qry_database_root, resize=resize_size)
	if long_trajectory:
		qry_database_root = "/storage/datasets/AerialLoc/Drone2Sat/GES/Test1_Ljubljana_150m_80fov_90deg_3000/"
		qry_image_dataset = GESQueryImages(root=qry_database_root, resize=resize_size)

	benchmark_top_k = [1, 5, 10, 25, 50, 100]
	benchmark_recall_at_xmeters = [100, 150, 200, 250, 500, 1000]

	save_dir = os.path.join(base_dir, "results/anyloc", time.strftime("%Y%m%d-%H%M%S"))
	os.makedirs(save_dir, exist_ok=True)

	# Pipeline
	preprocessor = AnyLocPreprocessor()
	extractor = DinoV2ExtractFeatures(dinov2_ckpt, layer=dinov2_desc_layer, facet=dinov2_desc_facet, device=device)
	if vlad_anyloc_prefitted:
		vlad_pre = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", device=device, domain="aerial").vlad
		vlad = VLAD(kmeans_type="fpk", minibatch=vlad_minibatch, num_clusters=vlad_pre.num_clusters, vlad_mode="hard", cache_dir=cache_dir, dist_mode="cosine", desc_dim=vlad_pre.desc_dim)
		vlad.kmeans = vlad_pre.kmeans
		vlad.kmeans.centroids = vlad_pre.kmeans.centroids.cpu()
		vlad.c_centers = vlad_pre.c_centers.cpu()
		del vlad_pre
	else:
		vlad = VLAD(kmeans_type=vlad_kmeans_type, minibatch=vlad_minibatch, num_clusters=vlad_num_clusters, vlad_mode=vlad_mode, cache_dir=cache_dir)

	rottraj_thetas = [0, 90, 180, 270] if ROTTRAJ_EXP else [0]
	rotexp_thetas = [0, 90, 180, 270] if ROTREF_EXP else [0]

	if not vlad_anyloc_prefitted:
		print("Fitting VLAD...")
		features = []
		for i, batch in enumerate(tqdm(ref_image_dataloader)):
			if DEBUG > 1 and i > 20:
				break
			if i % vlad_fit_step != 0:
				continue
			image = batch["image"]
			x = preprocessor(image, device=device)
			x = extractor(x)
			x = x.squeeze()
			x = x.flatten(start_dim=0, end_dim=-2)
			x = x.detach()
			x = x.cpu()

			if vlad.kmeans_type == "minibatch":
				vlad.fit(x)
			else:
				features.append(x)

		if vlad.kmeans_type != "minibatch":
			features = torch.stack(features)
			vlad.fit(x)

	if not load_faiss_index:
		print("Building database...")
		vdim = vlad.desc_dim * vlad.num_clusters
		db = VectorDatabase(vdim=vdim, distfn="l2", norm_vec=True)
		for theta in rotexp_thetas:
			for i, ref in enumerate(tqdm(ref_image_dataloader)):
				if DEBUG > 1 and i > 20:
					break
				image = ref["image"]
				image = F.rotate(image, theta)
				x = preprocessor(image, device=device)
				x = extractor(x)
				x = x.cpu()
				x = vlad.generate_multi(x)
				db.add(x)

		if save_faiss_index:
			print("Saving database...")
			db.save(save_dir)
	else:
		print("Loading database...")
		db = load_database(load_faiss_index_dir)

	print("Benchmarking...")
	N = len(qry_image_dataset)
	total_pipeline_time = 0.0
	total_db_search_time = 0.0
	max_memory_peak = 0.0
	max_vram_peak = 0.0

	tracemalloc.start()
	min_dists_at_k = np.zeros((N, len(benchmark_top_k)))
	recalls_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
	for theta in rottraj_thetas:
		for i, qry in enumerate(tqdm(qry_image_dataset)):
			if DEBUG > 1 and i > 20:
				break
			image = qry["image"]
			image = F.rotate(image, theta)

			pipeline_start_time = time.time()
			x = preprocessor(image, device=device)
			x = extractor(x)
			x = x.cpu()
			if VISUALIZE_VLAD:
				qry_residuals = vlad.generate_multi_res_vec(x)
			x = vlad.generate_multi(x)
			total_pipeline_time += time.time() - pipeline_start_time

			heatmap_benchmark_top_k = benchmark_top_k
			if PLOT_HEATMAP:
				heatmap_benchmark_top_k = [db.size()]

			db_search_start_time = time.time()
			dists, inds = db.search(x, max(heatmap_benchmark_top_k)) 	# Search database
			total_db_search_time += time.time() - db_search_start_time

			curr_iter_memory_peak = tracemalloc.get_traced_memory()[1] # in bytes
			curr_iter_vram_peak = torch.cuda.max_memory_allocated() # in bytes

			curr_iter_memory_peak = bytes_to_gb(curr_iter_memory_peak)
			curr_iter_vram_peak = bytes_to_gb(curr_iter_vram_peak)

			max_memory_peak = max(max_memory_peak, curr_iter_memory_peak)
			max_vram_peak = max(max_vram_peak, curr_iter_vram_peak)

			torch.cuda.reset_peak_memory_stats()
			tracemalloc.reset_peak()

			# Calculate distances
			coord_gt = [qry["lat"], qry["lon"]]
			coords_pr = []
			for cpr_i in range(max(benchmark_top_k)):
				ref = ref_image_dataset[inds[0, cpr_i] % len(ref_image_dataset)]
				coords_pr.append([
					ref["lat"],
					ref["lon"]
				])
			gdists = np.array([geopy.distance.geodesic(coord_gt, coord_pr).m for coord_pr in coords_pr])

			# Calculate top-k distances
			for j, k in enumerate(benchmark_top_k):
				min_dists_at_k[i, j] = np.min(gdists[:k])
			
			# Check if the top retrieved image is within a distance (recall@x meters)
			for j, xmeters in enumerate(benchmark_recall_at_xmeters):
				if min_dists_at_k[i, 0] <= xmeters:
					recalls_at_xmeters[j] += 1

			# Plot images and distances
			if PLOT:
				num_cols = 8
				fig, axes = plt.subplots(1, num_cols, figsize=(20, 4))
				axes[0].imshow(image.permute(1,2,0).cpu())
				axes[0].set_title(f"Qry | rot: {theta}°")
				axes[0].axis("off")

				for j in range(1, 6):
					ref_im_ix = inds[0, j-1] % len(ref_image_dataset)
					rot_ix = inds[0, j-1] // len(ref_image_dataset)
					rot = rotexp_thetas[rot_ix]
					ref_im = ref_image_dataset[ref_im_ix]["image"]
					ref_im = F.rotate(ref_im, rot)
					axes[j].imshow(ref_im.permute(1,2,0).cpu())
					axes[j].set_title(f"Ref{j-1} ({gdists[j-1]:.2f} m) | rot: {rot}°")
					axes[j].axis("off")

				fig.suptitle(f"Query #{i} theta: {theta} | min dist@1: {min_dists_at_k[i, 0]:.2f} m | min dist@5: {min_dists_at_k[i, 1]:.2f} m", fontsize=13)
				plt.tight_layout(rect=[0, 0, 1, 0.92])
				save_path = os.path.join(save_dir, f"query_{i:03d}_rot_{theta:03d}.png")

				if VISUALIZE_VLAD:
					# ax[8]
					colors = np.zeros((vlad.num_clusters, 3))
					# legend_lines = []
					# legend_nums = []
					for j in range(vlad.num_clusters):
						colors[j,:] = color_map_color(j/(vlad.num_clusters-1))
						# custom_line = Line2D([0], [0], color = color_map_color(
						# 		j/(vlad.num_clusters-1)), lw=4)
						# legend_lines.append(custom_line)
						# legend_nums.append(str(j))

					# Loop through all the patches inside image (Dino patches)
					# img_orig = F.resize(image, (64,64))
					img_desc = []
					for j in range(qry_residuals.shape[1]):
						cur_res_vec = torch.abs(qry_residuals[0, j])
						res_idx = torch.argmin(torch.sum(cur_res_vec, dim=1))
						img_desc.append(res_idx)
					img_desc = np.asarray(img_desc)
					# img_desc = np.reshape(np.asarray(img_desc), (64,64))

					# Color based on the closest clusters
					desc_color = np.zeros((img_desc.shape[0], 3))
					for c in range(vlad.num_clusters):
						img_idx = np.argwhere(img_desc==c)
						desc_color[img_idx[:,0]] = colors[c] * 255

					# Merge cluster color map with original image

					lh = lw = int(np.sqrt(img_desc.shape[0]))
					vlad_color_img = np.reshape(desc_color, (lh, lw, 3))
					vlad_color_img = cv2.resize(vlad_color_img, (image.shape[1], image.shape[2]), interpolation=cv2.INTER_NEAREST)

					axes[7].imshow(vlad_color_img.astype(np.uint8), alpha=0.7)
					axes[7].imshow(image.permute(1,2,0).cpu(), alpha=0.5)
					axes[7].set_title(f"VLAD Color Map")
					axes[7].axis("off")

				if PLOT_HEATMAP:
					# The tiles in the reference dataset are order by column top-to-bottom left-to-right
					heatmap_w = ref_image_dataset.metadata["bbox_area_size"]["width"]
					heatmap_h = ref_image_dataset.metadata["bbox_area_size"]["height"]
					heatmap = np.zeros((heatmap_h, heatmap_w))

					# Get the ground truth position
					ref_area_lu = ref_image_dataset.metadata["bbox_area_bounds"]["lu"]
					ref_area_rb = ref_image_dataset.metadata["bbox_area_bounds"]["rb"]

					ref_area_lu_lon = ref_area_lu["lon"]
					ref_area_lu_lat = ref_area_lu["lat"]
					ref_area_rb_lon = ref_area_rb["lon"]
					ref_area_rb_lat = ref_area_rb["lat"]

					# Calculate the ground truth position in the heatmap
					gt_row = int((qry["lat"] - ref_area_rb_lat) / (ref_area_lu_lat - ref_area_rb_lat) * heatmap_h)
					gt_col = int((qry["lon"] - ref_area_lu_lon) / (ref_area_rb_lon - ref_area_lu_lon) * heatmap_w)
					axes[6].plot(gt_col, gt_row, marker='x', color='green', markersize=10, markeredgewidth=2)

					for h, (dist, ind) in enumerate(zip(dists[0], inds[0])):
						if ROTREF_EXP:
							ind = ind % len(ref_image_dataset)
						# convert ind to position in heatmap
						row = ind % heatmap_h
						col = ind // heatmap_h
						heatmap[row, col] = dist
						if h == 0:
							axes[6].plot(col, row, marker='x', color='blue', markersize=10, markeredgewidth=2)
					
					# Normalize heatmap
					heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

					# Plot heatmap
					axes[6].imshow(heatmap, cmap="hot", interpolation="nearest")
					axes[6].set_title(f"Heatmap")
					axes[6].axis("off")

				plt.savefig(save_path)
				plt.close()

	tracemalloc.stop()

	print_file = os.path.join(save_dir, "results.txt")
	with open(print_file, "w") as file:
		avg_min_dist_at_k = np.mean(min_dists_at_k, axis=0)
		print(f"Avg. min distances at k (meters): {avg_min_dist_at_k}", file=file)

		# Calculate recalls
		recalls_at_xmeters = recalls_at_xmeters / (len(qry_image_dataset) * len(rottraj_thetas))
		for i, xmeters in enumerate(benchmark_recall_at_xmeters):
			print(f"Recall@1 at {xmeters} meters: {recalls_at_xmeters[i]:.4f}", file=file)
		
		# Calculate averages
		avg_pipeline_time = total_pipeline_time / (len(qry_image_dataset) * len(rottraj_thetas) + 1)
		avg_db_search_time = total_db_search_time / (len(qry_image_dataset) * len(rottraj_thetas) + 1)
		
		print(f"Avg. pipeline time: {avg_pipeline_time}", file=file)
		print(f"Avg. db search time: {avg_db_search_time}", file=file)
		print(f"Max memory peak: {max_memory_peak}", file=file)
		print(f"Max vram peak: {max_vram_peak}", file=file)
		print(f"Database size: {db.size()}", file=file)
		print(f"Vector dim: {db.vdim}", file=file)


if __name__ == "__main__":
	main()



#*############## Results ##############*#

""" Experiment 1:

Avg. min distances at k (meters): [447.16186842 184.42492005]
Avg. pipeline time: 2.4638406685420446
Avg. db search time: 0.006488902228219169
Max memory peak: 0.00022271554917097092
Max vram peak: 4.806108474731445
Database size: 108
Vector dim: 49152

"""

"""Experiment 2 (256x256 reference images (resized to 512x512))
Avg. min distances at k (meters): [317.57166427  76.88561577]
Avg. pipeline time: 0.5160617283412389
Avg. db search time: 0.12261880465916225
Max memory peak: 0.20031056366860867
Max vram peak: 4.407312393188477
Database size: 1813
Vector dim: 49152

"""