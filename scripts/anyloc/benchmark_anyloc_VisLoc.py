# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python scripts/anyloc/benchmark_anyloc_GURS.py --vdbdir /storage/datasets/AerialLoc/Drone2Sat/GURSSat/vdbs/test_vdb --traj_config configs/anyloc/benchmark_anyloc_GURS.yaml

import os
import time
import json
import torch
import argparse
import tracemalloc
import numpy as np
from tqdm import tqdm

import torchvision.transforms.functional as TF

from geoloc.eval.utils import bytes_to_gb
from geoloc.visualize import visualize_top_k_retrieved, visualize_vlad_clusters, visualize_similarity_heatmap_visloc
from geoloc.data.vdb import load_database
from geoloc.config_parser import load_config, class_from_config
from geoloc.utils import DEBUG
from geoloc.eval.metrics import calculate_distances, calculate_intersections
from geoloc.eval.utils import write_resdict_to_file, write_pretty_table


def create_argparse():
	parser = argparse.ArgumentParser(description="AnyLoc Benchmarking")
	parser.add_argument("--vdbdir", type=str, help="Path to the vector database directory")
	parser.add_argument("--traj_config", type=str, help="Path to the trajectory config file")
	return parser


@torch.no_grad()
def benchmark_anyloc_visloc(vdbdir, traj_config):

	config = load_config(
		os.path.join(vdbdir, "config.yaml"))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	loaddir = config.savedir # savedir from build_vdb_anyloc is now loaddir
	savedir = os.path.join(loaddir, "benchmark")
	os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)

	ref_image_dataset = class_from_config(config.dataset)
	qry_image_dataset = class_from_config(traj_config.dataset)

	anyloc = class_from_config(config.anyloc).to(device)
	anyloc.load_vlad(loaddir)
	anyloc.load_pca(loaddir)

	benchmark_top_k = traj_config.benchmark_top_k
	benchmark_recall_at_xmeters = traj_config.benchmark_recall_at_xmeters

	rotexp_thetas = [0, 90, 180, 270] if config.ROTREF_EXP else [0]
	rottraj_thetas = [0, 90, 180, 270] if traj_config.ROTTRAJ_EXP else [0]
	if traj_config.ROTTRAJ_EXP > 0:
		print("Running rotation trajectory experiment...")

	print("Loading Vector Database...")
	db = load_database(vdbdir, config)

	print("Benchmarking...")
	pipeline_times = []
	db_search_times = []
	memory_peaks = []
	vram_peaks = []

	tracemalloc.start()
	all_min_dists_at_k = []
	recalls_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
	intersection_recalls_at_k = np.zeros(len(benchmark_top_k))
	num_qry_images = 0
	predicted_trajectory = {}
	for theta in rottraj_thetas:
		for i, qry in enumerate(tqdm(qry_image_dataset)):
			if DEBUG > 1 and i > 10:
				break
			if i % traj_config.every_n != 0:
				continue
			num_qry_images += 1
			image = qry["image"].to(device)
			image = TF.rotate(image, theta)

			pipeline_start_time = time.time()
			x = anyloc(image.unsqueeze(0), return_residuals=traj_config.VISUALIZE_VLAD) # Add batch dimension before calling anyloc
			if traj_config.VISUALIZE_VLAD:
				x, qry_residuals = x
			pipeline_times.append(time.time() - pipeline_start_time)

			heatmap_benchmark_top_k = benchmark_top_k
			if traj_config.VISUALIZE_HEATMAP:
				heatmap_benchmark_top_k = [db.size()]

			db_search_start_time = time.time()
			dists, inds = db.search(x, max(heatmap_benchmark_top_k))
			db_search_times.append(time.time() - db_search_start_time)

			curr_iter_memory_peak = bytes_to_gb(
				tracemalloc.get_traced_memory()[1])
			curr_iter_vram_peak = bytes_to_gb(
				torch.cuda.max_memory_allocated())
			
			memory_peaks.append(curr_iter_memory_peak)
			vram_peaks.append(curr_iter_vram_peak)

			torch.cuda.reset_peak_memory_stats()
			tracemalloc.reset_peak()

			# Calculate distances
			gdists, ref_points = calculate_distances(
				qry, benchmark_top_k, inds, qry_image_dataset, ref_image_dataset)

			# Save predicted trajectory
			predicted_trajectory[i] = {
				"lon": ref_points[0][0],
				"lat": ref_points[0][1],
			}

			# Calculate top-k distances
			min_dists_at_k = np.zeros(len(benchmark_top_k))
			for j, k in enumerate(benchmark_top_k):
				min_dists_at_k[j] = np.min(gdists[:k])
			all_min_dists_at_k.append(min_dists_at_k)

			# Calculate intersections
			gt_pos, intersection_tps = calculate_intersections(
				qry, benchmark_top_k, inds, ref_image_dataset)

			for j, k in enumerate(benchmark_top_k):
				if np.any(intersection_tps[:k]):
					intersection_recalls_at_k[j] += 1
				
			# Check if the top retrieved image is within a distance (recall@x meters)
			for j, xmeters in enumerate(benchmark_recall_at_xmeters):
				if min_dists_at_k[0] <= xmeters:
					recalls_at_xmeters[j] += 1

			# Plot images and distances
			if traj_config.VISUALIZE_TOP_K:
				visualize_top_k_retrieved(
					query_image=image, query_ix=i, inds=inds, 
					ref_image_dataset=ref_image_dataset, savedir=savedir, 
					gdists=gdists, min_dists_at_k=min_dists_at_k,
					rotexp_thetas=rotexp_thetas, query_theta=theta, gt_pos=gt_pos
				)
			if traj_config.VISUALIZE_VLAD:
				visualize_vlad_clusters(
					query_image=image, query_ix=i, query_residuals=qry_residuals,
					vlad=anyloc.vlad, savedir=savedir, query_theta=theta
				)
			if traj_config.VISUALIZE_HEATMAP:
				visualize_similarity_heatmap_visloc(
					dists=dists, inds=inds, ref_image_dataset=ref_image_dataset, 
					query_ix=i, query_theta=theta, 
					savedir=savedir
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
	with open(print_file, "a") as file:
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

		all_min_dists_at_k = np.array(all_min_dists_at_k)
		avg_min_dist_at_k = np.mean(all_min_dists_at_k, axis=0)
		avg_min_dist_data = [
			[k, avg_min_dist_at_k[i]] for i, k in enumerate(benchmark_top_k)
		]
		write_pretty_table(
			file=file,
			field_names=["Top-k", "Avg. min distance (m)"],
			data=avg_min_dist_data,
			align="l",
			header="Avg. Min Distance"
		)

		recalls_at_xmeters = recalls_at_xmeters / num_qry_images
		recall_data = [
			[xmeters, recalls_at_xmeters[i]] for i, xmeters in enumerate(benchmark_recall_at_xmeters)
		]
		write_pretty_table(
			file=file,
			field_names=["Distance (m)", "Recall@1"],
			data=recall_data,
			align="l",
			header="Distance Recalls"
		)

		intersection_recalls_at_k = intersection_recalls_at_k / num_qry_images
		intersection_recall_data = [
			[k, intersection_recalls_at_k[i]] for i, k in enumerate(benchmark_top_k)
		]
		write_pretty_table(
			file=file,
			field_names=["Top-k", "Intersection Recall"],
			data=intersection_recall_data,
			align="l",
			header="Intersection Recalls"
		)


def main():
	parser = create_argparse()
	args = parser.parse_args()
	traj_config = load_config(args.traj_config)
	benchmark_anyloc_visloc(args.vdbdir, traj_config)


if __name__ == "__main__":
	main()