"""
This file contains the code and results of the experiments for the AnyLoc method on the VPAir dataset.
"""
import os
import sys
research_dir = f"{os.path.dirname(os.path.realpath(__file__))}/../"
sys.path.insert(1, research_dir)

import os
from natsort import natsorted
import torchvision
import torch
from typing import List, Union
import numpy as np
import einops as ein
import sklearn.cluster
import fast_pytorch_kmeans as fpk
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
import tracemalloc
import faiss
import time

from AnyLoc.utilities import DinoV2ExtractFeatures
from research.data.dataset_VPAir import VPAirReferenceImages, VPAirQueryImages
from research.data.vdb import VectorDatabase
from research.aggregators.vlad import VLAD

#*############## Global arguments ##############*#

DEBUG = int(os.environ.get("DEBUG", 0))

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

	# Local Arguments
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	dinov2_ckpt = "dinov2_vitg14"
	dinov2_desc_layer = 31
	dinov2_desc_facet = "value"

	vlad_num_clusters = 32
	vlad_mode = "hard"
	cache_dir = None
	vlad_minibatch = 1024
	vlad_fit_step = 1
	vlad_kmeans_type = "minibatch"

	database_root = "/storage/datasets/AerialLoc/Drone2Sat/VPAir"
	vpair_include_distractors = True
	soft_positive_offset = 3

	benchmark_top_k = [1, 5]

	if vpair_include_distractors:
		assert vlad_kmeans_type == "minibatch", "KMeans type must be minibatch for distractors, for RAM efficiency"

	# Databases
	ref_image_dataset = VPAirReferenceImages(root=database_root, include_distractors=vpair_include_distractors)
	qry_image_dataset = VPAirQueryImages(root=database_root, soft_positive_offset=soft_positive_offset)

	# Pipeline
	preprocessor = AnyLocPreprocessor()
	extractor = DinoV2ExtractFeatures(dinov2_ckpt, layer=dinov2_desc_layer, facet=dinov2_desc_facet, device=device)
	vlad = VLAD(kmeans_type=vlad_kmeans_type, minibatch=vlad_minibatch, num_clusters=vlad_num_clusters, vlad_mode=vlad_mode, cache_dir=cache_dir)

	print("Fitting VLAD...")
	features = []
	for i in tqdm(range(0, len(ref_image_dataset), vlad_fit_step)):
		if DEBUG > 1 and i > 20:
			break
		r = ref_image_dataset[i]
		image = r["image"]
		x = preprocessor(image, device=device)
		x = extractor(x)
		x = x.squeeze()
		x = x.flatten(start_dim=1)
		x = x.detach()
		x = x.cpu()

		if vlad.kmeans_type == "minibatch":
			vlad.fit(x)
		else:
			features.append(x)

	if vlad_kmeans_type != "minibatch":
		features = torch.stack(features)
		vlad.fit(x)

	dummy_agg_feats = vlad(x.unsqueeze(0))

	print("Building database...")
	db = VectorDatabase(vdim=dummy_agg_feats.shape[1], distfn="l2", use_gpu=-1, norm_vec=True)
	for i, ref in enumerate(tqdm(ref_image_dataset)):
		if DEBUG > 1 and i > 20:
			break
		image = ref["image"]
		x = preprocessor(image, device=device)
		x = extractor(x)
		x = vlad(x)
		db.add(x)


	print("Benchmarking...")
	N = len(qry_image_dataset)
	recalls = dict(zip(benchmark_top_k, [0]*len(benchmark_top_k)))
	total_pipeline_time = 0.0
	total_db_search_time = 0.0
	max_memory_peak = 0.0
	max_vram_peak = 0.0

	tracemalloc.start()
	for i, qry in enumerate(tqdm(qry_image_dataset)):
		if DEBUG > 1 and i > 20:
			break
		image = qry["image"]

		pipeline_start_time = time.time()
		x = preprocessor(image, device=device)
		x = extractor(x)
		x = vlad(x)
		total_pipeline_time += time.time() - pipeline_start_time

		db_search_start_time = time.time()
		dists, inds = db.search(x, max(benchmark_top_k)) 	# Search database
		total_db_search_time += time.time() - db_search_start_time

		curr_iter_memory_peak = tracemalloc.get_traced_memory()[1] # in bytes
		curr_iter_vram_peak = torch.cuda.max_memory_allocated() # in bytes

		curr_iter_memory_peak = bytes_to_gb(curr_iter_memory_peak)
		curr_iter_vram_peak = bytes_to_gb(curr_iter_vram_peak)

		max_memory_peak = max(max_memory_peak, curr_iter_memory_peak)
		max_vram_peak = max(max_vram_peak, curr_iter_vram_peak)

		torch.cuda.reset_peak_memory_stats()
		tracemalloc.reset_peak()

		# Calculate recalls
		gt_pos = qry["gt_pos"]
		for k in benchmark_top_k:
			if np.any(np.isin(inds[0, :k], gt_pos)):
				recalls[k] += 1

	tracemalloc.stop()

	# Calculate averages
	avg_pipeline_time = total_pipeline_time / N
	avg_db_search_time = total_db_search_time / N

	# Calculate percentage recalls
	for k in recalls:
		recalls[k] /= N

	# Display results
	print("Recalls:")
	for k, v in recalls.items():
		print(f"Recall @ {k}: {v}")
	
	print(f"Avg. pipeline time: {avg_pipeline_time}")
	print(f"Avg. db search time: {avg_db_search_time}")
	print(f"Max memory peak: {max_memory_peak}")
	print(f"Max vram peak: {max_vram_peak}")
	print(f"Database size: {db.size()}")
	print(f"Vector dim: {db.vdim}")


if __name__ == "__main__":
	main()



#*############## Results ##############*#


""" Experiment 1:

No distractors, 
DINOv2 Giant, 
value facet, 
layer 31, 
Minibatch KMeans

Recall @ 1: 				0.5946045824094605
Recall @ 5: 				0.7261640798226164
Avg. pipeline time: 		0.8907155364157091
Avg. db search time: 		0.10075180220410107
Max memory peak: 			0.00023200269788503647
Max vram peak: 				4.51662540435791
Database size: 				2706
Vector dim: 				49152
"""



""" Experiment 2:

with distractors, 
DINOv2 Giant, v
alue facet, 
layer 31, 
Minibatch KMeans

Recall @ 1: 				0.5140428677014043
Recall @ 5: 				0.6341463414634146
Avg. pipeline time: 		0.820732752393989
Avg. db search time: 		0.26315268708967055
Max memory peak: 			0.0002331119030714035
Max vram peak: 				4.51662540435791
Database size: 				12706
Vector dim: 				49152

"""
