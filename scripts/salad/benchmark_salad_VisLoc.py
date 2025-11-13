# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. DEBUG=2 python scripts/salad/benchmark_salad_VisLoc.py --vdbdir /storage/datasets/AerialLoc/Drone2Sat/vdbs/salad/VisLoc/paper_no_distractors --traj_config configs/salad/VisLoc/benchmark_salad_VisLoc.yaml

import os
import time
import json
import torch
import argparse
import tracemalloc
import numpy as np
from tqdm import tqdm

from geoloc.eval.utils import bytes_to_gb
from geoloc.visualize import (
    visualize_similarity_heatmap_visloc,
    visualize_top_k_retrieved,
)
from geoloc.data.vdb import load_database
from geoloc.config_parser import load_config, class_from_config
from geoloc.utils import DEBUG
from geoloc.eval.utils import write_resdict_to_file, write_pretty_table
from geoloc.eval.metrics import calculate_distances, calculate_intersections


def create_argparse():
    parser = argparse.ArgumentParser(description="SALAD Benchmarking")
    parser.add_argument(
        "--vdbdir", type=str, help="Path to the vector database directory"
    )
    parser.add_argument(
        "--traj_config", type=str, help="Path to the trajectory config file"
    )
    return parser


@torch.no_grad()
def benchmark_salad_visloc(vdbdir, traj_config):

    config = load_config(os.path.join(vdbdir, "config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaddir = config.savedir  # savedir from build_vdb_salad is now loaddir
    savedir = os.path.join(loaddir, "benchmark")
    os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)

    ref_image_dataset = class_from_config(config.dataset)
    qry_image_dataset = class_from_config(traj_config.dataset)

    salad_cls, _ = class_from_config(config.salad, instantiate=False)
    salad = salad_cls.load_from_checkpoint(config.salad.checkpoint, strict=True).to(
        device
    )
    salad.eval()

    benchmark_top_k = traj_config.benchmark_top_k
    benchmark_recall_at_xmeters = traj_config.benchmark_recall_at_xmeters

    print("Loading Vector Database...")
    db = load_database(vdbdir, config)

    print("Benchmarking...")
    pipeline_times = []
    db_search_times = []
    memory_peaks = []
    vram_peaks = []

    tracemalloc.start()
    # recalls = np.zeros(len(benchmark_top_k))
    all_min_dists_at_k = []
    recalls_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
    intersection_recalls_at_k = np.zeros(len(benchmark_top_k))
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
        x = salad(image.unsqueeze(0))  # Add batch dimension before calling salad
        pipeline_times.append(time.time() - pipeline_start_time)

        db_search_start_time = time.time()
        dists, inds = db.search(x, max(benchmark_top_k))
        db_search_times.append(time.time() - db_search_start_time)

        curr_iter_memory_peak = bytes_to_gb(tracemalloc.get_traced_memory()[1])
        curr_iter_vram_peak = bytes_to_gb(torch.cuda.max_memory_allocated())

        memory_peaks.append(curr_iter_memory_peak)
        vram_peaks.append(curr_iter_vram_peak)

        torch.cuda.reset_peak_memory_stats()
        tracemalloc.reset_peak()

        # Calculate distances
        # gdists, ref_points = calculate_distances(
        # 	qry, benchmark_top_k, inds, qry_image_dataset, ref_image_dataset)

        # Save predicted trajectory
        # predicted_trajectory[i] = {
        # 	"lon": ref_points[0][0],
        # 	"lat": ref_points[0][1],
        # }

        # Calculate top-k distances
        # min_dists_at_k = np.zeros(len(benchmark_top_k))
        # for j, k in enumerate(benchmark_top_k):
        # 	min_dists_at_k[j] = np.min(gdists[:k])
        # all_min_dists_at_k.append(min_dists_at_k)

        # Calculate intersections
        gt_pos, intersection_tps = calculate_intersections(
            qry, benchmark_top_k, inds, ref_image_dataset
        )

        for j, k in enumerate(benchmark_top_k):
            if np.any(intersection_tps[:k]):
                intersection_recalls_at_k[j] += 1

        # # Check if the top retrieved image is within a distance (recall@x meters)
        # for j, xmeters in enumerate(benchmark_recall_at_xmeters):
        # 	if min_dists_at_k[0] <= xmeters:
        # 		recalls_at_xmeters[j] += 1

        # Calculate recalls
        # for j, k in enumerate(benchmark_top_k):
        # 	inds_k = inds[0, :k] % len(ref_image_dataset)
        # 	if np.any(np.isin(inds_k, gt_pos)):
        # 		intersection_recalls_at_k[j] += 1

        # Plot images and distances
        if traj_config.VISUALIZE_TOP_K:
            visualize_top_k_retrieved(
                query_image=image,
                query_ix=i,
                inds=inds,
                ref_image_dataset=ref_image_dataset,
                savedir=savedir,
                gt_pos=gt_pos,
            )
        if traj_config.VISUALIZE_HEATMAP:
            visualize_similarity_heatmap_visloc(
                dists=dists,
                inds=inds,
                ref_image_dataset=ref_image_dataset,
                query_ix=i,
                query_theta=0.0,
                savedir=savedir,
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
        intersection_recalls_at_k = intersection_recalls_at_k / num_qry_images
        intersection_recall_data = [
            [k, intersection_recalls_at_k[i]] for i, k in enumerate(benchmark_top_k)
        ]
        write_pretty_table(
            file=file,
            field_names=["Top-k", "Recall"],
            data=intersection_recall_data,
            align="l",
            header="Recall",
        )


def main():
    parser = create_argparse()
    args = parser.parse_args()
    traj_config = load_config(args.traj_config)
    benchmark_salad_visloc(args.vdbdir, traj_config)


if __name__ == "__main__":
    main()
