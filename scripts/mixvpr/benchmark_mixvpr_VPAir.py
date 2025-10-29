# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. DEBUG=2 python scripts/mixvpr/benchmark_mixvpr_VPAir.py --vdbdir /storage/datasets/AerialLoc/Drone2Sat/vdbs/mixvpr/VPAir/paper_no_distractors --traj_config configs/mixvpr/VPAir/benchmark_mixvpr_VPAir.yaml

import os
import time
import json
import torch
import argparse
import tracemalloc
import numpy as np
from tqdm import tqdm

from geoloc.eval.utils import bytes_to_gb
from geoloc.visualize import visualize_top_k_retrieved, visualize_vlad_clusters
from geoloc.data.vdb import load_database
from geoloc.config_parser import load_config, class_from_config
from geoloc.utils import DEBUG
from geoloc.eval.utils import write_resdict_to_file, write_pretty_table


def create_argparse():
	parser = argparse.ArgumentParser(description="mixvpr Benchmarking")
	parser.add_argument("--vdbdir", type=str, help="Path to the vector database directory")
	parser.add_argument("--traj_config", type=str, help="Path to the trajectory config file")
	return parser

@torch.no_grad()
def benchmark_mixvpr_vpair(vdbdir, traj_config):

	config = load_config(
		os.path.join(vdbdir, "config.yaml"))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	loaddir = config.savedir # savedir from build_vdb_mixvpr is now loaddir
	savedir = os.path.join(loaddir, "benchmark")
	os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)

	ref_image_dataset = class_from_config(config.dataset)
	qry_image_dataset = class_from_config(traj_config.dataset)

	mixvpr_cls, _ = class_from_config(config.mixvpr, instantiate=False)
	mixvpr = mixvpr_cls.load_from_checkpoint(config.mixvpr.checkpoint, strict=True).to(device)
	mixvpr.eval()

	benchmark_top_k = traj_config.benchmark_top_k

	print("Loading Vector Database...")
	db = load_database(vdbdir, config)

	print("Benchmarking...")
	pipeline_times = []
	db_search_times = []
	memory_peaks = []
	vram_peaks = []

	tracemalloc.start()
	recalls = np.zeros(len(benchmark_top_k))
	num_qry_images = 0
	predicted_trajectory = {}
	for i, qry in enumerate(tqdm(qry_image_dataset)):
		if DEBUG > 1 and i > 10:
			break
		if i % traj_config.every_n != 0:
			continue
		num_qry_images += 1
		image = qry["image"].to(device)

		pipeline_start_time = time.time()
		x = mixvpr(image.unsqueeze(0)) # Add batch dimension before calling mixvpr
		pipeline_times.append(time.time() - pipeline_start_time)

		db_search_start_time = time.time()
		dists, inds = db.search(x, max(benchmark_top_k))
		db_search_times.append(time.time() - db_search_start_time)

		curr_iter_memory_peak = bytes_to_gb(
			tracemalloc.get_traced_memory()[1])
		curr_iter_vram_peak = bytes_to_gb(
			torch.cuda.max_memory_allocated())
		
		memory_peaks.append(curr_iter_memory_peak)
		vram_peaks.append(curr_iter_vram_peak)

		torch.cuda.reset_peak_memory_stats()
		tracemalloc.reset_peak()

		# Save predicted trajectory
		ref_point = ref_image_dataset[(inds[0, 0] % len(ref_image_dataset)).item()]
		predicted_trajectory[i] = {
			"lon": ref_point["lon"],
			"lat": ref_point["lat"],
		}

		# Calculate recalls
		gt_pos = qry["gt_pos"]
		for j, k in enumerate(benchmark_top_k):
			inds_k = inds[0, :k] % len(ref_image_dataset)
			if np.any(np.isin(inds_k, gt_pos)):
				recalls[j] += 1

		# Plot images and distances
		if traj_config.VISUALIZE_TOP_K:
			visualize_top_k_retrieved(
				query_image=image, query_ix=i, inds=inds, 
				ref_image_dataset=ref_image_dataset, 
				savedir=savedir, gt_pos=gt_pos
			)
	tracemalloc.stop()

	avg_pipeline_time = np.mean(pipeline_times)
	avg_db_search_time = np.mean(db_search_times)
	max_memory_peak = np.max(memory_peaks)
	max_vram_peak = np.max(vram_peaks)

	# save predicted trajectory to json file
	with open(os.path.join(savedir, f"predicted_trajectory.json"), "w") as f:
		json.dump(predicted_trajectory, f, indent=4)

	print_file = os.path.join(savedir, "results.txt")
	with open(print_file, "w") as file:
		resdict_vdb = {
			"Vec. Database size": db.size(),
			"Vector dimension": db.vdim,
			"Num. query images": num_qry_images,
		}
		write_resdict_to_file(file, resdict_vdb, header="Metadata")

		resdict_perf = {
			"Avg. pipeline time": f"{avg_pipeline_time:.4f} s",
			"Avg. db search time": f"{avg_db_search_time:.4f} s",
			"Max memory peak": f"{max_memory_peak:.4f} GB",
			"Max vram peak": f"{max_vram_peak:.4f} GB",
		}
		write_resdict_to_file(file, resdict_perf, header="Performance")
		recalls = recalls / num_qry_images
		recall_data = [
			[k, recalls[i]] for i, k in enumerate(benchmark_top_k)
		]
		write_pretty_table(
			file=file,
			field_names=["Top-k", "Recall"],
			data=recall_data,
			align="l",
			header="Recall"
		)


def main():
	parser = create_argparse()
	args = parser.parse_args()
	traj_config = load_config(args.traj_config)
	benchmark_mixvpr_vpair(args.vdbdir, traj_config)


if __name__ == "__main__":
	main()