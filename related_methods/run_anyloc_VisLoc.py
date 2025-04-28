"""
This file contains the code and results of the experiments for the AnyLoc method on the VisLoc dataset.
"""
import os
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

from AnyLoc.utilities import DinoV2ExtractFeatures
from research.data.vdb import VectorDatabase, load_database
from research.data.dataset_VisLoc import VisLocReferenceImages, VisLocQueryImages
from research.aggregators.vlad import VLAD

#*############## Global arguments ##############*#

DEBUG = int(os.environ.get("DEBUG", 0))
PLOT = int(os.environ.get("PLOT", 0))
PLOT_HEATMAP = int(os.environ.get("PLOT_HEATMAP", 0))

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
	vlad_fit_batchsize = 16
	vlad_anyloc_prefitted = True

	save_faiss_index = False
	load_faiss_index = False
	load_faiss_index_dir = None

	base_dir = "/storage/datasets/AerialLoc/UAV_VisLoc"
	
	flight_idx = 1
	tile_size = 1024
	stride = 512
	resize_size = (256, 256)

	ref_image_dataset = VisLocReferenceImages(root=base_dir, flight_idx=flight_idx, tile_size=tile_size, stride=stride, resize=resize_size)
	ref_image_dataloader = DataLoader(ref_image_dataset, batch_size=vlad_fit_batchsize, shuffle=False)
	qry_image_dataset = VisLocQueryImages(root=base_dir, flight_idx=flight_idx, resize=resize_size)

	# benchmark_top_k = [1, 5, 10, 25, 50, 100]
	benchmark_top_k = [1, 5, 10, 25, 50]
	benchmark_recall_at_xmeters = [100, 150, 200, 250, 500, 1000]

	save_dir = os.path.join(base_dir, f"{flight_idx:02d}/results/anyloc", time.strftime("%Y%m%d-%H%M%S"))
	os.makedirs(save_dir, exist_ok=True)

	# Pipeline
	preprocessor = AnyLocPreprocessor()
	extractor = DinoV2ExtractFeatures(dinov2_ckpt, layer=dinov2_desc_layer, facet=dinov2_desc_facet, device=device)
	if vlad_anyloc_prefitted:
		vlad = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", device=device, domain="aerial").vlad
		vlad.c_centers = vlad.c_centers.cpu()
	else:
		vlad = VLAD(kmeans_type=vlad_kmeans_type, minibatch=vlad_minibatch, num_clusters=vlad_num_clusters, vlad_mode=vlad_mode, cache_dir=cache_dir)

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
		for i, ref in enumerate(tqdm(ref_image_dataloader)):
		# for i, ref in enumerate(tqdm(ref_image_dataset)):
			if DEBUG > 1 and i > 20:
				break
			image = ref["image"]
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
	for i, qry in enumerate(tqdm(qry_image_dataset)):
		if DEBUG > 1 and i > 20:
			break
		image = qry["image"]

		pipeline_start_time = time.time()
		x = preprocessor(image, device=device)
		x = extractor(x)
		x = x.cpu()
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
		coords_pr = [
			[ref_image_dataset[inds[0, i]]["lat"], ref_image_dataset[inds[0, i]]["lon"]] for i in range(max(benchmark_top_k))
		]
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
			num_cols = 7 if PLOT_HEATMAP else 6
			fig, axes = plt.subplots(1, num_cols, figsize=(18, 4))
			image = torch.clamp(image.permute(1,2,0).cpu(), 0.0, 1.0)
			axes[0].imshow(image)
			axes[0].set_title("Query Image")
			axes[0].axis("off")

			for j in range(1, 6):
				img_show = ref_image_dataset[inds[0, j-1]]["image"].permute(1,2,0).cpu()
				img_show = torch.clamp(img_show, 0.0, 1.0)
				axes[j].imshow(img_show)
				axes[j].set_title(f"Ref Image {j-1} ({gdists[j-1]:.2f} m)")
				axes[j].axis("off")

			fig.suptitle(f"Query #{i} | min dist@1: {min_dists_at_k[i, 0]:.2f} m | min dist@5: {min_dists_at_k[i, 1]:.2f} m", fontsize=13)
			plt.tight_layout(rect=[0, 0, 1, 0.92])
			save_path = os.path.join(save_dir, f"query_{i:03d}.png")

			# plt.savefig(save_path)
			# plt.close()

			if PLOT_HEATMAP:
				# The tiles in the reference dataset are order by column top-to-bottom left-to-right
				heatmap_w = ref_image_dataset.num_tiles_width
				heatmap_h = ref_image_dataset.num_tiles_height
				heatmap = np.zeros((heatmap_h, heatmap_w))

				for i, (dist, ind) in enumerate(zip(dists[0], inds[0])):
					# convert ind to position in heatmap
					row = ind % heatmap_h
					col = ind // heatmap_h
					heatmap[row, col] = dist
					if i == 0:
						axes[6].plot(col, row, marker='x', color='cyan', markersize=10, markeredgewidth=2)
				
				# Normalize heatmap
				heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
				heatmap = np.clip(heatmap, 0.0, 1.0)  # Ensure values are in [0, 1]
				# Plot heatmap
				axes[6].imshow(heatmap, cmap="hot", interpolation="nearest")
				axes[6].axis("off")
				axes[6].set_title("Heatmap")
				# plt.savefig(os.path.join(save_dir, f"heatmap_{i:03d}.jpg"))
				plt.savefig(save_path)
				plt.close()

	tracemalloc.stop()
	
	print_file = os.path.join(save_dir, "results.txt")
	with open(print_file, "w") as file:
		avg_min_dist_at_k = np.mean(min_dists_at_k, axis=0)
		print(f"Avg. min distances at k (meters): {avg_min_dist_at_k}", file=file)

		# Calculate recalls
		recalls_at_xmeters = recalls_at_xmeters / (i+1)
		for i, xmeters in enumerate(benchmark_recall_at_xmeters):
			print(f"Recall@1 at {xmeters} meters: {recalls_at_xmeters[i]:.4f}", file=file)
		
		# Calculate averages
		avg_pipeline_time = total_pipeline_time / (i+1)
		avg_db_search_time = total_db_search_time / (i+1)
		
		print(f"Avg. pipeline time: {avg_pipeline_time}", file=file)
		print(f"Avg. db search time: {avg_db_search_time}", file=file)
		print(f"Max memory peak: {max_memory_peak}", file=file)
		print(f"Max vram peak: {max_vram_peak}", file=file)
		print(f"Database size: {db.size()}", file=file)
		print(f"Vector dim: {db.vdim}", file=file)


if __name__ == "__main__":
	main()



#*############## Results ##############*#