# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. DEBUG=2 python scripts/benchmark.py --vdbdir /path/to/vdb --traj_config configs/model/dataset/benchmark_config.yaml

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
from geoloc.visualize import (
    visualize_top_k_retrieved,
    visualize_vlad_clusters,
    visualize_similarity_heatmap_visloc,
    visualize_similarity_heatmap_gurs,
)
from geoloc.data.vdb import load_database
from geoloc.config_parser import load_config, save_config, class_from_config
from geoloc.utils import DEBUG, load_model, get_model_type, get_dataset_type, requires_arg
from geoloc.eval.metrics import calculate_distances, calculate_intersections, hit_at_k, tp_at_k, safe_rank1, rank1_or_sentinel, reciprocal_rank_from_rank1, average_precision
from geoloc.eval.utils import write_resdict_to_file, write_pretty_table, segvlad_get_matches

INDEX_BASED_DATASETS = ["vpair", "alto", "ortholoc"]
DISTANCE_BASED_DATASETS = ["visloc", "gurs"]

def create_argparse():
    parser = argparse.ArgumentParser(
        description="Benchmarking script"
    )
    parser.add_argument(
        "--vdbdir", type=str, required=True, help="Path to the vector database directory"
    )
    parser.add_argument(
        "--traj_config", type=str, required=True, help="Path to the trajectory config file"
    )
    return parser

def get_savedir(loaddir: str):
    savedir = os.path.join(loaddir, "benchmark")
    counter = 1
    while os.path.exists(savedir):
        savedir = os.path.join(loaddir, f"benchmark{counter}")
        counter += 1
    return savedir

def supports_vlad_visualization(model_type: str) -> bool:
    return model_type in ["AnyLoc"]

def get_position_keys(dataset_type: str) -> tuple:
    if dataset_type in ["alto", "gurs"]:
        return ("east", "north")
    return ("lon", "lat")

def benchmark_main(vdbdir: str, traj_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_config = load_config(os.path.join(vdbdir, "build_config.yaml"))
    model = load_model(build_config).eval().to(device)

    ref_image_dataset = class_from_config(build_config.dataset)
    qry_image_dataset = class_from_config(traj_config.dataset)

    model_type = get_model_type(model)
    dataset_type = get_dataset_type(qry_image_dataset)
    print(f"Detected model: {model_type}")
    print(f"Detected dataset: {dataset_type}")

    # Distance-based metrics (for VisLoc, GURS)
    is_distance_based = dataset_type in DISTANCE_BASED_DATASETS
    lon_key, lat_key = get_position_keys(dataset_type)

    savedir = get_savedir(build_config.savedir)
    os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)

    visualize_vlad = getattr(traj_config, "VISUALIZE_VLAD", False) and supports_vlad_visualization(model_type)
    visualize_top_k = getattr(traj_config, "VISUALIZE_TOP_K", False)
    visualize_heatmap = getattr(traj_config, "VISUALIZE_HEATMAP", False)
    save_salad_matrix = getattr(traj_config, "SAVE_SALAD_MATRIX", False)
    rotexp_thetas = [0, 90, 180, 270] if getattr(build_config, "ROTREF_EXP", False) else [0]
    rottraj_thetas = [0, 90, 180, 270] if getattr(traj_config, "ROTTRAJ_EXP", False) else [0]
    benchmark_recall_at_xmeters = getattr(traj_config, "benchmark_recall_at_xmeters", [])
    benchmark_top_k = traj_config.benchmark_top_k
    every_n = traj_config.every_n

    if getattr(traj_config, "ROTTRAJ_EXP", False):
        print("Running rotation trajectory experiment...")

    print(f"Loading Vector Database for {model_type}...")
    vdb = load_database(vdbdir, build_config)

    benchmark_results = benchmark_loop(
        model=model, model_type=model_type, vdb=vdb, qry_image_dataset=qry_image_dataset,
        ref_image_dataset=ref_image_dataset, vdbdir=vdbdir, dataset_type=dataset_type, is_distance_based=is_distance_based,
        lon_key=lon_key, lat_key=lat_key, benchmark_recall_at_xmeters=benchmark_recall_at_xmeters, 
        benchmark_top_k=benchmark_top_k, rotexp_thetas=rotexp_thetas, rottraj_thetas=rottraj_thetas, 
        visualize_vlad=visualize_vlad, visualize_top_k=visualize_top_k, visualize_heatmap=visualize_heatmap, 
        every_n=every_n, savedir=savedir, device=device, save_salad_matrix=save_salad_matrix
    )

    num_qry_images = benchmark_results["num_qry_images"]
    # predicted_trajectory = benchmark_results["predicted_trajectory"]
    results_metastats = benchmark_results["results_metastats"]
    avg_pipeline_time = benchmark_results["avg_pipeline_time"]
    avg_db_search_time = benchmark_results["avg_db_search_time"]
    max_memory_peak = benchmark_results["max_memory_peak"]
    max_vram_peak = benchmark_results["max_vram_peak"]
    recalls_at_k = benchmark_results["recalls_at_k"]
    intersection_recalls_at_k = benchmark_results["intersection_recalls_at_k"]
    recalls_at_xmeters = benchmark_results["recalls_at_xmeters"]
    avg_min_dist_at_k = benchmark_results["avg_min_dists_at_k"]
    mean_reciprocal_rank = benchmark_results["mean_reciprocal_rank"]
    mean_average_precision = benchmark_results["mean_average_precision"]
    # is_distance_based = benchmark_results["is_distance_based"]
    
    # Save rotator thetas if available
    # rotator_thetas_path = os.path.join(savedir, "rotator_thetas.npy")
    # rotator_thetas = benchmark_results.get("rotator_thetas", None)
    # if rotator_thetas is not None:
    #     np.save(rotator_thetas_path, rotator_thetas)
    
    # Write results
    print_file = os.path.join(savedir, "results.txt")
    with open(print_file, "w") as file:
        resdict_vdb = {
            "Model": model_type,
            "Dataset": dataset_type,
            "Vec. Database size": vdb.size(),
            "Vector dimension": vdb.vdim,
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
        
        if is_distance_based:
            # Distance-based metrics
            # all_min_dists_at_k = np.array(all_min_dists_at_k)
            # avg_min_dist_at_k = np.mean(all_min_dists_at_k, axis=0)
            avg_min_dist_data = [
                [k, avg_min_dist_at_k[i]] for i, k in enumerate(benchmark_top_k)
            ]
            write_pretty_table(
                file=file,
                field_names=["Top-k", "Avg. min distance (m)"],
                data=avg_min_dist_data,
                align="l",
                header="Avg. Min Distance",
            )
            
            # recalls_at_xmeters = recalls_at_xmeters / num_qry_images
            recall_data = [
                [xmeters, recalls_at_xmeters[i]]
                for i, xmeters in enumerate(benchmark_recall_at_xmeters)
            ]
            write_pretty_table(
                file=file,
                field_names=["Distance (m)", "Recall@1"],
                data=recall_data,
                align="l",
                header="Distance Recalls",
            )
            
            # intersection_recalls_at_k = intersection_recalls_at_k / num_qry_images
            intersection_recall_data = [
                [k, intersection_recalls_at_k[i]] for i, k in enumerate(benchmark_top_k)
            ]
            write_pretty_table(
                file=file,
                field_names=["Top-k", "Intersection Recall"],
                data=intersection_recall_data,
                align="l",
                header="Intersection Recalls",
            )
        else:
            # Index-based metrics
            # recalls = recalls / num_qry_images
            recall_data = [[k, recalls_at_k[i]] for i, k in enumerate(benchmark_top_k)]
            write_pretty_table(
                file=file,
                field_names=["Top-k", "Recall"],
                data=recall_data,
                align="l",
                header="Recall",
            )

    # Save metastats
    with open(os.path.join(savedir, "results_metastats.json"), "w") as f:
        json.dump(results_metastats, f)

    print(f"Results saved to {print_file}")
    save_config(traj_config, savedir, prefix="benchmark")

@torch.no_grad()
def benchmark_loop(
    model, model_type, vdb, qry_image_dataset, ref_image_dataset, 
    dataset_type,  vdbdir, is_distance_based=None, lon_key=None, lat_key=None, 
    benchmark_recall_at_xmeters=[100, 250, 500, 1000], 
    benchmark_top_k=[1, 5, 10, 25, 50, 100], rotexp_thetas=[0], rottraj_thetas=[0], 
    visualize_vlad=False, visualize_top_k=False, visualize_heatmap=False, every_n=1, 
    savedir=None, device="cpu", save_salad_matrix=False,
):
    
    if is_distance_based is None:
        is_distance_based = dataset_type in DISTANCE_BASED_DATASETS
    if is_distance_based:
        lon_key, lat_key = get_position_keys(dataset_type)

    print(f"Benchmarking {model_type} on {dataset_type}...")
    pipeline_times = []
    db_search_times = []
    memory_peaks = []
    vram_peaks = []

    all_min_dists_at_k = None
    recalls_at_xmeters = None
    intersection_recalls_at_k = None
    recalls_at_k = None
    avg_min_dist_at_k = None
    rotator_thetas = []

    all_top10_inds = []
    all_tps_at_k = []
    all_hits_at_k = []
    all_rank1 = []
    all_rank1_sortable = []
    all_reciprocal_rank = []
    all_average_precision = []

    if is_distance_based:
        all_min_dists_at_k = []
        recalls_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
        intersection_recalls_at_k = np.zeros(len(benchmark_top_k))
    else:
        recalls_at_k = np.zeros(len(benchmark_top_k))
    
    num_qry_images = 0
    # predicted_trajectory = {}
    results_metastats = {}

    tracemalloc.start()
    for j, theta in enumerate(rottraj_thetas):
        for i, qry in enumerate(tqdm(qry_image_dataset)):
            if DEBUG > 1 and i > 10:
                break
            if i % every_n != 0:
                continue
            num_qry_images += 1

            benchmark_results = benchmark_single(
                model=model, model_type=model_type, vdb=vdb, qry_image_dataset=qry_image_dataset,
                ref_image_dataset=ref_image_dataset, vdbdir=vdbdir, dataset_type=dataset_type, query_ix=i,
                is_distance_based=is_distance_based, lon_key=lon_key, lat_key=lat_key, theta=theta, theta_ix=j,
                benchmark_recall_at_xmeters=benchmark_recall_at_xmeters, benchmark_top_k=benchmark_top_k,
                rotexp_thetas=rotexp_thetas, visualize_vlad=visualize_vlad, visualize_top_k=visualize_top_k,
                visualize_heatmap=visualize_heatmap, savedir=savedir, device=device, save_salad_matrix=save_salad_matrix,
            )
            # predicted_trajectory[i] = benchmark_results["predicted_coordinates"]
            if is_distance_based:
                all_min_dists_at_k.append(benchmark_results["min_dists_at_k"])
                intersection_recalls_at_k += benchmark_results["intersection_recall_at_k"]
                recalls_at_xmeters += benchmark_results["recall_at_xmeters"]
            else:
                recalls_at_k += benchmark_results["recall_at_k"]

            all_top10_inds.append(benchmark_results["top10_inds"])
            all_tps_at_k.append(benchmark_results["tps_at_k"])
            all_hits_at_k.append(benchmark_results["hits_at_k"])
            all_rank1.append(benchmark_results["rank1"])
            all_rank1_sortable.append(benchmark_results["rank1_sortable"])
            all_reciprocal_rank.append(benchmark_results["reciprocal_rank"])
            all_average_precision.append(benchmark_results["average_precision"])

            pipeline_times.append(benchmark_results["pipeline_time"])
            db_search_times.append(benchmark_results["db_search_time"])
            memory_peaks.append(benchmark_results["curr_iter_memory_peak"])
            vram_peaks.append(benchmark_results["curr_iter_vram_peak"])
            rotator_theta = benchmark_results.get("rotator_theta", None)


            results_metastats[i] = {
                "predicted_coordinates": benchmark_results["predicted_coordinates"],
                "recall_at_k": benchmark_results["recall_at_k"].tolist() if benchmark_results["recall_at_k"] is not None else None,
                "recall_at_xmeters": benchmark_results["recall_at_xmeters"].tolist() if benchmark_results["recall_at_xmeters"] is not None else None,
                "intersection_recall_at_k": benchmark_results["intersection_recall_at_k"].tolist() if benchmark_results["intersection_recall_at_k"] is not None else None,
                "min_dists_at_k": benchmark_results["min_dists_at_k"].tolist() if benchmark_results["min_dists_at_k"] is not None else None,
                "pipeline_time": benchmark_results["pipeline_time"],
                "db_search_time": benchmark_results["db_search_time"],
                "curr_iter_memory_peak": benchmark_results["curr_iter_memory_peak"],
                "curr_iter_vram_peak": benchmark_results["curr_iter_vram_peak"],
                "top10_inds": benchmark_results["top10_inds"],
                "tps_at_k": benchmark_results["tps_at_k"].tolist(),
                "hits_at_k": benchmark_results["hits_at_k"].tolist(),
                "rank1": benchmark_results["rank1"],
                "rank1_sortable": benchmark_results["rank1_sortable"],
                "reciprocal_rank": benchmark_results["reciprocal_rank"],
                "average_precision": benchmark_results["average_precision"],
                "rotator_theta": rotator_theta,
            }
            # results_metastats[i]["predicted_coordinates"] = benchmark_results["predicted_coordinates"]
            # results_metastats[i]["recall_at_k"] = benchmark_results["recall_at_k"]
            # results_metastats[i]["recall_at_xmeters"] = benchmark_results["recall_at_xmeters"]
            # results_metastats[i]["intersection_recall_at_k"] = benchmark_results["intersection_recall_at_k"]
            # results_metastats[i]["min_dists_at_k"] = benchmark_results["min_dists_at_k"]
            # results_metastats[i]["pipeline_time"] = benchmark_results["pipeline_time"]
            # results_metastats[i]["db_search_time"] = benchmark_results["db_search_time"]
            # results_metastats[i]["curr_iter_memory_peak"] = benchmark_results["curr_iter_memory_peak"]
            # results_metastats[i]["curr_iter_vram_peak"] = benchmark_results["curr_iter_vram_peak"]
            # results_metastats[i]["top10_inds"] = benchmark_results["top10_inds"]
            # results_metastats[i]["tps_at_k"] = benchmark_results["tps_at_k"]
            # results_metastats[i]["hits_at_k"] = benchmark_results["hits_at_k"]
            # results_metastats[i]["rank1"] = benchmark_results["rank1"]
            # results_metastats[i]["rank1_sortable"] = benchmark_results["rank1_sortable"]
            # results_metastats[i]["reciprocal_rank"] = benchmark_results["reciprocal_rank"]
            # results_metastats[i]["average_precision"] = benchmark_results["average_precision"]
            # results_metastats[i]["rotator_theta"] = rotator_theta

            if rotator_theta is not None:
                rotator_thetas.append(rotator_theta)

            torch.cuda.reset_peak_memory_stats()
            tracemalloc.reset_peak()
    tracemalloc.stop()

    avg_pipeline_time = np.mean(pipeline_times)
    avg_db_search_time = np.mean(db_search_times)
    max_memory_peak = np.max(memory_peaks)
    max_vram_peak = np.max(vram_peaks)
    rotator_thetas = np.array(rotator_thetas) if len(rotator_thetas) > 0 else np.array([])

    if is_distance_based:
        all_min_dists_at_k = np.array(all_min_dists_at_k)
        avg_min_dist_at_k = np.mean(all_min_dists_at_k, axis=0)
        recalls_at_xmeters = recalls_at_xmeters / num_qry_images
        intersection_recalls_at_k = intersection_recalls_at_k / num_qry_images
    else:
        recalls_at_k = recalls_at_k / num_qry_images

    mean_reciprocal_rank = np.mean(all_reciprocal_rank)
    mean_ap = np.mean(all_average_precision)

    results_metastats["num_qry_images"] = num_qry_images
    results_metastats["avg_pipeline_time"] = avg_pipeline_time
    results_metastats["avg_db_search_time"] = avg_db_search_time
    results_metastats["max_memory_peak"] = max_memory_peak
    results_metastats["max_vram_peak"] = max_vram_peak
    results_metastats["is_distance_based"] = is_distance_based
    results_metastats["recalls_at_k"] = recalls_at_k.tolist() if recalls_at_k is not None else None
    results_metastats["recalls_at_xmeters"] = recalls_at_xmeters.tolist() if recalls_at_xmeters is not None else None
    results_metastats["avg_min_dist_at_k"] = avg_min_dist_at_k.tolist() if avg_min_dist_at_k is not None else None
    results_metastats["mean_reciprocal_rank"] = mean_reciprocal_rank
    results_metastats["mean_average_precision"] = mean_ap

    return dict(
        num_qry_images=num_qry_images,
        # predicted_trajectory=predicted_trajectory,
        results_metastats=results_metastats,
        avg_pipeline_time=avg_pipeline_time,
        avg_db_search_time=avg_db_search_time,
        max_memory_peak=max_memory_peak,
        max_vram_peak=max_vram_peak,
        recalls_at_k=recalls_at_k,
        intersection_recalls_at_k=intersection_recalls_at_k,
        recalls_at_xmeters=recalls_at_xmeters,
        avg_min_dists_at_k=avg_min_dist_at_k,
        rotator_thetas=rotator_thetas,
        mean_reciprocal_rank=mean_reciprocal_rank,
        mean_average_precision=mean_ap,
    )

@torch.no_grad()
def benchmark_single(
    model, model_type, vdb, qry_image_dataset, ref_image_dataset, vdbdir, dataset_type, query_ix,
    is_distance_based=None, lon_key=None, lat_key=None, theta=0, theta_ix=0, benchmark_recall_at_xmeters=[100, 250, 500, 1000],
    benchmark_top_k=[1, 5, 10, 25, 50, 100], rotexp_thetas=[0], visualize_vlad=False, 
    visualize_top_k=False, visualize_heatmap=False, savedir=None, device=None, save_salad_matrix=False
):
    if is_distance_based is None:
        is_distance_based = dataset_type in DISTANCE_BASED_DATASETS
    if lon_key is None or lat_key is None:
        lon_key, lat_key = get_position_keys(dataset_type)

    # num_qry_images += 1
    qry = qry_image_dataset[query_ix]
    image = qry["image"]
    if device is not None:
        image = image.to(device)

    if theta != 0:
        image = TF.rotate(image, theta)
    
    model_args = {}
    if visualize_vlad:
        model_args["return_residuals"] = True
    if save_salad_matrix:
        model_args["return_salad_matrix"] = True
    # if model_type in ["SegVLAD", "Mast3rRetrievalModel"]:
    if requires_arg(model.forward, "idx"):
        model_args["idx"] = query_ix + theta_ix * len(qry_image_dataset)

    pipeline_start_time = time.time()
    outdict = model(image.unsqueeze(0), **model_args)
    pipeline_time = time.time() - pipeline_start_time

    x = outdict["out"]
    x_rotator_theta = outdict.get("theta", None)
    if x_rotator_theta is not None:
        x_rotator_theta = x_rotator_theta.item()

    qry_residuals = outdict.get("qry_residuals", None)

    salad_matrix = outdict.get("salad_matrix", None)
    if salad_matrix is not None:
        salad_matrix = salad_matrix.cpu().detach()
    
    search_top_k = [vdb.size()] if visualize_heatmap else benchmark_top_k

    search_args = {}
    if model_type in ["Mast3rRetrievalModel"]:
        search_args["imids"] = outdict["ids"].cpu().numpy()

    db_search_start_time = time.time()
    dists, inds = vdb.search(qu=x, k=max(search_top_k), **search_args)
    db_search_time = time.time() - db_search_start_time

    if model_type in ["SegVLAD"]:
        if len(vdb.imInd_buffer) > 0: # necessary for validation during training
            ref_imInds = vdb.imInd_buffer
        else:
            ref_imInds = torch.load(os.path.join(vdbdir, "segvlad_imInds.npy")).numpy()
        # pred, pred_score = segvlad_get_matches(dists, inds, ref_imInds, n=max(benchmark_top_k))
        inds, dists = segvlad_get_matches(dists, inds, ref_imInds, n=max(benchmark_top_k)) # !! this is not distances, just accumulated similarities
    
    curr_iter_memory_peak = bytes_to_gb(tracemalloc.get_traced_memory()[1])
    curr_iter_vram_peak = bytes_to_gb(torch.cuda.max_memory_allocated())

    # torch.cuda.reset_peak_memory_stats()
    # tracemalloc.reset_peak()
    
    predicted_coordinate = None
    recall_at_k = None
    recall_at_xmeters = None
    intersection_recall_at_k = None
    min_dists_at_k = None

    if is_distance_based:
        gdists, ref_points = calculate_distances(
            qry, benchmark_top_k, inds, qry_image_dataset, ref_image_dataset
        )
        predicted_coordinate = {
            lon_key: ref_points[0][0],
            lat_key: ref_points[0][1],
        }
        # predicted_trajectory[query_ix] = {
        #     lon_key: ref_points[0][0],
        #     lat_key: ref_points[0][1],
        # }
        
        # Calculate top-k distances
        min_dists_at_k = np.zeros(len(benchmark_top_k))
        for h, k in enumerate(benchmark_top_k):
            min_dists_at_k[h] = np.min(gdists[:k])
        # all_min_dists_at_k.append(min_dists_at_k)
        
        # Calculate intersections
        gt_pos, intersection_tps = calculate_intersections(
            qry, benchmark_top_k, inds, ref_image_dataset
        )
        tp_flags = np.array(intersection_tps, dtype=bool)

        # --- Metrics ---
        hits_at_k = np.array([hit_at_k(tp_flags, k) for k in benchmark_top_k], dtype=int)
        tps_at_k  = np.array([tp_at_k(tp_flags,  k) for k in benchmark_top_k], dtype=int)
        recall_at_k = hits_at_k.astype(float)

        rank1 = safe_rank1(tp_flags)
        rank1_sortable = rank1_or_sentinel(tp_flags, 999)
        rr = reciprocal_rank_from_rank1(rank1)
        ap = average_precision(tp_flags)

        # Recall@x meters (currently checks top-1 only in your code)
        recall_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
        for h, xmeters in enumerate(benchmark_recall_at_xmeters):
            if min_dists_at_k[0] <= xmeters:
                recall_at_xmeters[h] += 1
        # If you instead want "any in top-K within x", use:
        # if np.any(gdists[:K_main] <= xmeters): recall_at_xmeters[h] += 1

        
        # intersection_recall_at_k = np.zeros(len(benchmark_top_k))
        # for h, k in enumerate(benchmark_top_k):
        #     if np.any(intersection_tps[:k]):
        #         intersection_recall_at_k[h] += 1
        
        # # Recall@x meters
        # recall_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
        # for h, xmeters in enumerate(benchmark_recall_at_xmeters):
        #     if min_dists_at_k[0] <= xmeters:
        #         recall_at_xmeters[h] += 1
        #         # recalls_at_xmeters[h] += 1
    else:
        ref_point = ref_image_dataset[(inds[0, 0] % len(ref_image_dataset)).item()]
        predicted_coordinate = {
            lon_key: ref_point[lon_key],
            lat_key: ref_point[lat_key],
        }
        # predicted_trajectory[query_ix] = {
        #     lon_key: ref_point[lon_key],
        #     lat_key: ref_point[lat_key],
        # }
        
        gt_pos = qry["gt_pos"]
        retrieved = (inds[0, :] % len(ref_image_dataset)).astype(int)
        gt_set = set(gt_pos if isinstance(gt_pos, (list, tuple, np.ndarray)) else [gt_pos])

        # TP flags from index membership
        tp_flags = np.array([r in gt_set for r in retrieved], dtype=bool)

        # Your "recall_at_k" is Hit@K
        hits_at_k = np.array([hit_at_k(tp_flags, k) for k in benchmark_top_k], dtype=int)
        recall_at_k = hits_at_k.astype(float)

        # Additional metrics
        tps_at_k  = np.array([tp_at_k(tp_flags,  k) for k in benchmark_top_k], dtype=int)
        rank1 = safe_rank1(tp_flags)
        rank1_sortable = rank1_or_sentinel(tp_flags, 999)
        rr = reciprocal_rank_from_rank1(rank1)
        ap = average_precision(tp_flags)

        # recall_at_k = np.zeros(len(benchmark_top_k))
        # for h, k in enumerate(benchmark_top_k):
        #     inds_k = inds[0, :k] % len(ref_image_dataset)
        #     if np.any(np.isin(inds_k, gt_pos)):
        #         recall_at_k[h] += 1
        #         # recalls[h] += 1
    
    # Visualizations
    K_main = 10
    # hitK = hit_at_k(tp_flags, K_main)
    tpK  = tp_at_k(tp_flags, K_main)
    ap10000 = int(round(ap * 10000))
    query_id = qry.get("id", f"q{query_ix:05d}")
    filename = (
        f"AP{ap10000:05d}_R{rank1_sortable:03d}_TP@{K_main}-{tpK:02d}_{query_id}.png"
    )
    if visualize_top_k:
        rotator_ref_thetas_path = os.path.join(vdbdir, f"thetas.npy")
        rotator_ref_thetas = None
        if os.path.exists(rotator_ref_thetas_path):
            rotator_ref_thetas = np.load(rotator_ref_thetas_path).reshape(-1).tolist()

        vis_kwargs = {
            "query_image": image,
            "query_ix": query_ix,
            "inds": inds,
            "ref_image_dataset": ref_image_dataset,
            "savedir": savedir,
            "rotator_theta": x_rotator_theta,
            "rotator_ref_thetas": rotator_ref_thetas,
            "filename": filename,
        }
        if is_distance_based:
            vis_kwargs["gdists"] = gdists
            vis_kwargs["min_dists_at_k"] = min_dists_at_k
            vis_kwargs["rotexp_thetas"] = rotexp_thetas
            vis_kwargs["query_theta"] = theta
            vis_kwargs["gt_pos"] = gt_pos
        else:
            vis_kwargs["gt_pos"] = qry["gt_pos"]
        visualize_top_k_retrieved(**vis_kwargs)
    
    if visualize_vlad and qry_residuals is not None:
        vlad_module = getattr(model, "vlad", None)
        if vlad_module is not None:
            visualize_vlad_clusters(
                query_image=image,
                query_ix=query_ix,
                query_residuals=qry_residuals,
                vlad=vlad_module,
                savedir=savedir,
                query_theta=theta,
            )
    
    if visualize_heatmap:
        if dataset_type == "VisLoc":
            visualize_similarity_heatmap_visloc(
                dists=dists,
                inds=inds,
                ref_image_dataset=ref_image_dataset,
                query_ix=query_ix,
                query_theta=theta,
                savedir=savedir,
            )
        elif dataset_type == "GURS":
            qry_east, qry_north = qry["lon"], qry["lat"]
            visualize_similarity_heatmap_gurs(
                dists=dists,
                inds=inds,
                ref_image_dataset=ref_image_dataset,
                query_ix=query_ix,
                query_theta=theta,
                query_northing=qry_north,
                query_easting=qry_east,
                savedir=savedir,
                # rotref_exp=getattr(build_config, "ROTREF_EXP", False),
                rotref_exp=True if rotexp_thetas != [0] else False,
            )
    if save_salad_matrix and salad_matrix is not None:
        salad_savedir = os.path.join(savedir, "qry_salad_matrices")
        salad_filename = filename.replace(".png", ".npy")
        os.makedirs(salad_savedir, exist_ok=True)
        salad_path = os.path.join(salad_savedir, salad_filename)
        np.save(salad_path, salad_matrix.cpu().numpy())

    return dict(
        pipeline_time=pipeline_time,
        db_search_time=db_search_time,
        curr_iter_memory_peak=curr_iter_memory_peak,
        curr_iter_vram_peak=curr_iter_vram_peak,
        predicted_coordinates=predicted_coordinate,
        recall_at_k=recall_at_k,
        recall_at_xmeters=recall_at_xmeters,
        intersection_recall_at_k=intersection_recall_at_k,
        min_dists_at_k=min_dists_at_k,
        rotator_theta=x_rotator_theta,
        top10_inds=inds[0, :10].tolist(), # Save index of top-10 retrieved images
        tps_at_k=tps_at_k,
        hits_at_k=hits_at_k,
        rank1=rank1,
        rank1_sortable=rank1_sortable,
        reciprocal_rank=rr,
        average_precision=ap,
    )






# @torch.no_grad()
# def benchmark(
#     model, model_type, vdb, qry_image_dataset, ref_image_dataset, 
#     dataset_type, benchmark_recall_at_xmeters=[100, 250, 500, 1000], 
#     benchmark_top_k=[1, 5, 10, 25, 50, 100], rotexp_thetas=[0], rottraj_thetas=[0], 
#     visualize_vlad=False, visualize_top_k=False, visualize_heatmap=False, every_n=1, 
#     savedir=None, device="cpu"
# ):
#     # Distance-based metrics (for VisLoc, GURS)
#     is_distance_based = dataset_type in DISTANCE_BASED_DATASETS
#     lon_key, lat_key = get_position_keys(dataset_type)
    
#     print(f"Benchmarking {model_type} on {dataset_type}...")
#     pipeline_times = []
#     db_search_times = []
#     memory_peaks = []
#     vram_peaks = []
    
#     tracemalloc.start()
    
#     if is_distance_based:
#         all_min_dists_at_k = []
#         recalls_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
#         intersection_recalls_at_k = np.zeros(len(benchmark_top_k))
#     else:
#         recalls = np.zeros(len(benchmark_top_k))
    
#     num_qry_images = 0
#     predicted_trajectory = {}
#     for j, theta in enumerate(rottraj_thetas):
#         for i, qry in enumerate(tqdm(qry_image_dataset)):
#             if DEBUG > 1 and i > 10:
#                 break
#             if i % every_n != 0:
#                 continue
#             num_qry_images += 1
#             image = qry["image"].to(device)
            
#             if theta != 0:
#                 image = TF.rotate(image, theta)
            
#             model_args = {}
#             if visualize_vlad:
#                 model_args["return_residuals"] = True
#             if model_type in ["SegVLAD", "Mast3rRetrievalModel"]:
#                 model_args["idx"] = i + j * len(qry_image_dataset)

#             pipeline_start_time = time.time()
#             outdict = model(image.unsqueeze(0), **model_args)
#             pipeline_times.append(time.time() - pipeline_start_time)

#             x = outdict["out"]
#             qry_residuals = outdict.get("qry_residuals", None)
            
#             search_top_k = [vdb.size()] if visualize_heatmap else benchmark_top_k

#             search_args = {}
#             if model_type in ["SegVLAD", "Mast3rRetrievalModel"]:
#                 search_args["imids"] = outdict["ids"].cpu().numpy()
#             db_search_start_time = time.time()
#             dists, inds = vdb.search(qu=x, k=max(search_top_k), **search_args)
#             db_search_times.append(time.time() - db_search_start_time)
            
#             curr_iter_memory_peak = bytes_to_gb(tracemalloc.get_traced_memory()[1])
#             curr_iter_vram_peak = bytes_to_gb(torch.cuda.max_memory_allocated())
            
#             memory_peaks.append(curr_iter_memory_peak)
#             vram_peaks.append(curr_iter_vram_peak)
            
#             torch.cuda.reset_peak_memory_stats()
#             tracemalloc.reset_peak()
            
#             if is_distance_based:
#                 gdists, ref_points = calculate_distances(
#                     qry, benchmark_top_k, inds, qry_image_dataset, ref_image_dataset
#                 )
#                 predicted_trajectory[i] = {
#                     lon_key: ref_points[0][0],
#                     lat_key: ref_points[0][1],
#                 }
                
#                 # Calculate top-k distances
#                 min_dists_at_k = np.zeros(len(benchmark_top_k))
#                 for j, k in enumerate(benchmark_top_k):
#                     min_dists_at_k[j] = np.min(gdists[:k])
#                 all_min_dists_at_k.append(min_dists_at_k)
                
#                 # Calculate intersections
#                 gt_pos, intersection_tps = calculate_intersections(
#                     qry, benchmark_top_k, inds, ref_image_dataset
#                 )
                
#                 for j, k in enumerate(benchmark_top_k):
#                     if np.any(intersection_tps[:k]):
#                         intersection_recalls_at_k[j] += 1
                
#                 # Recall@x meters
#                 for j, xmeters in enumerate(benchmark_recall_at_xmeters):
#                     if min_dists_at_k[0] <= xmeters:
#                         recalls_at_xmeters[j] += 1
#             else:
#                 ref_point = ref_image_dataset[(inds[0, 0] % len(ref_image_dataset)).item()]
#                 predicted_trajectory[i] = {
#                     lon_key: ref_point[lon_key],
#                     lat_key: ref_point[lat_key],
#                 }
                
#                 gt_pos = qry["gt_pos"]
#                 for j, k in enumerate(benchmark_top_k):
#                     inds_k = inds[0, :k] % len(ref_image_dataset)
#                     if np.any(np.isin(inds_k, gt_pos)):
#                         recalls[j] += 1
            
#             # Visualizations
#             if visualize_top_k:
#                 vis_kwargs = {
#                     "query_image": image,
#                     "query_ix": i,
#                     "inds": inds,
#                     "ref_image_dataset": ref_image_dataset,
#                     "savedir": savedir,
#                 }
#                 if is_distance_based:
#                     vis_kwargs["gdists"] = gdists
#                     vis_kwargs["min_dists_at_k"] = min_dists_at_k
#                     vis_kwargs["rotexp_thetas"] = rotexp_thetas
#                     vis_kwargs["query_theta"] = theta
#                     vis_kwargs["gt_pos"] = gt_pos
#                 else:
#                     vis_kwargs["gt_pos"] = qry["gt_pos"]
#                 visualize_top_k_retrieved(**vis_kwargs)
            
#             if visualize_vlad and qry_residuals is not None:
#                 vlad_module = getattr(model, "vlad", None)
#                 if vlad_module is not None:
#                     visualize_vlad_clusters(
#                         query_image=image,
#                         query_ix=i,
#                         query_residuals=qry_residuals,
#                         vlad=vlad_module,
#                         savedir=savedir,
#                         query_theta=theta,
#                     )
            
#             if visualize_heatmap:
#                 if dataset_type == "VisLoc":
#                     visualize_similarity_heatmap_visloc(
#                         dists=dists,
#                         inds=inds,
#                         ref_image_dataset=ref_image_dataset,
#                         query_ix=i,
#                         query_theta=theta,
#                         savedir=savedir,
#                     )
#                 elif dataset_type == "GURS":
#                     qry_east, qry_north = qry["lon"], qry["lat"]
#                     visualize_similarity_heatmap_gurs(
#                         dists=dists,
#                         inds=inds,
#                         ref_image_dataset=ref_image_dataset,
#                         query_ix=i,
#                         query_theta=theta,
#                         query_northing=qry_north,
#                         query_easting=qry_east,
#                         savedir=savedir,
#                         # rotref_exp=getattr(build_config, "ROTREF_EXP", False),
#                         rotref_exp=True if rotexp_thetas != [0] else False,
#                     )
#     tracemalloc.stop()
#     avg_pipeline_time = np.mean(pipeline_times)
#     avg_db_search_time = np.mean(db_search_times)
#     max_memory_peak = np.max(memory_peaks)
#     max_vram_peak = np.max(vram_peaks)

#     return dict(
#         num_qry_images=num_qry_images,
#         predicted_trajectory=predicted_trajectory,
#         avg_pipeline_time=avg_pipeline_time,
#         avg_db_search_time=avg_db_search_time,
#         max_memory_peak=max_memory_peak,
#         max_vram_peak=max_vram_peak,
#         recalls=recalls if not is_distance_based else intersection_recalls_at_k,
#         recalls_at_xmeters=recalls_at_xmeters if is_distance_based else None,
#         all_min_dists_at_k=all_min_dists_at_k if is_distance_based else None,
#         is_distance_based=is_distance_based,
#     )








# @torch.no_grad()
# def benchmark(vdbdir: str, traj_config):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     build_config = load_config(os.path.join(vdbdir, "build_config.yaml"))
#     model = load_model(build_config).eval().to(device)

#     ref_image_dataset = class_from_config(build_config.dataset)
#     qry_image_dataset = class_from_config(traj_config.dataset)

#     model_type = model.__class__.__name__
#     dataset_type = traj_config.dataset.class_path.split(".")[-2]
    
#     savedir = get_savedir(build_config.savedir)
#     os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)

#     print(f"Detected model: {model_type}")
#     print(f"Detected dataset: {dataset_type}")
    
#     benchmark_top_k = traj_config.benchmark_top_k

#     visualize_vlad = getattr(traj_config, "VISUALIZE_VLAD", False) and supports_vlad_visualization(model_type)
#     visualize_top_k = getattr(traj_config, "VISUALIZE_TOP_K", False)
#     visualize_heatmap = getattr(traj_config, "VISUALIZE_HEATMAP", False)
    
#     # Distance-based metrics (for VisLoc, GURS)
#     is_distance_based = dataset_type in DISTANCE_BASED_DATASETS
#     benchmark_recall_at_xmeters = getattr(traj_config, "benchmark_recall_at_xmeters", [])
    
#     rotexp_thetas = [0, 90, 180, 270] if getattr(build_config, "ROTREF_EXP", False) else [0]
#     rottraj_thetas = [0, 90, 180, 270] if getattr(traj_config, "ROTTRAJ_EXP", False) else [0]
#     if getattr(traj_config, "ROTTRAJ_EXP", False):
#         print("Running rotation trajectory experiment...")
    
#     lon_key, lat_key = get_position_keys(dataset_type)
    
#     print(f"Loading Vector Database for {model_type}...")
#     db = load_database(vdbdir, build_config)
    
#     print(f"Benchmarking {model_type} on {dataset_type}...")
#     pipeline_times = []
#     db_search_times = []
#     memory_peaks = []
#     vram_peaks = []
    
#     tracemalloc.start()
    
#     if is_distance_based:
#         all_min_dists_at_k = []
#         recalls_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
#         intersection_recalls_at_k = np.zeros(len(benchmark_top_k))
#     else:
#         recalls = np.zeros(len(benchmark_top_k))
    
#     num_qry_images = 0
#     predicted_trajectory = {}
    
#     for j, theta in enumerate(rottraj_thetas):
#         for i, qry in enumerate(tqdm(qry_image_dataset)):
#             if DEBUG > 1 and i > 10:
#                 break
#             if i % traj_config.every_n != 0:
#                 continue
#             num_qry_images += 1
#             image = qry["image"].to(device)
            
#             if theta != 0:
#                 image = TF.rotate(image, theta)
            
#             model_args = {}
#             if visualize_vlad:
#                 model_args["return_residuals"] = True
#             if model_type in ["SegVLAD", "Mast3rRetrievalModel"]:
#                 model_args["idx"] = i + j * len(qry_image_dataset)

#             pipeline_start_time = time.time()
#             outdict = model(image.unsqueeze(0), **model_args)
#             pipeline_times.append(time.time() - pipeline_start_time)

#             x = outdict["out"]
#             qry_residuals = outdict.get("qry_residuals", None)
            
#             search_top_k = [db.size()] if visualize_heatmap else benchmark_top_k

#             search_args = {}
#             if model_type in ["SegVLAD", "Mast3rRetrievalModel"]:
#                 search_args["imids"] = outdict["ids"].cpu().numpy()
#             db_search_start_time = time.time()
#             dists, inds = db.search(qu=x, k=max(search_top_k), **search_args)
#             db_search_times.append(time.time() - db_search_start_time)
            
#             curr_iter_memory_peak = bytes_to_gb(tracemalloc.get_traced_memory()[1])
#             curr_iter_vram_peak = bytes_to_gb(torch.cuda.max_memory_allocated())
            
#             memory_peaks.append(curr_iter_memory_peak)
#             vram_peaks.append(curr_iter_vram_peak)
            
#             torch.cuda.reset_peak_memory_stats()
#             tracemalloc.reset_peak()
            
#             if is_distance_based:
#                 gdists, ref_points = calculate_distances(
#                     qry, benchmark_top_k, inds, qry_image_dataset, ref_image_dataset
#                 )
#                 predicted_trajectory[i] = {
#                     lon_key: ref_points[0][0],
#                     lat_key: ref_points[0][1],
#                 }
                
#                 # Calculate top-k distances
#                 min_dists_at_k = np.zeros(len(benchmark_top_k))
#                 for j, k in enumerate(benchmark_top_k):
#                     min_dists_at_k[j] = np.min(gdists[:k])
#                 all_min_dists_at_k.append(min_dists_at_k)
                
#                 # Calculate intersections
#                 gt_pos, intersection_tps = calculate_intersections(
#                     qry, benchmark_top_k, inds, ref_image_dataset
#                 )
                
#                 for j, k in enumerate(benchmark_top_k):
#                     if np.any(intersection_tps[:k]):
#                         intersection_recalls_at_k[j] += 1
                
#                 # Recall@x meters
#                 for j, xmeters in enumerate(benchmark_recall_at_xmeters):
#                     if min_dists_at_k[0] <= xmeters:
#                         recalls_at_xmeters[j] += 1
#             else:
#                 ref_point = ref_image_dataset[(inds[0, 0] % len(ref_image_dataset)).item()]
#                 predicted_trajectory[i] = {
#                     lon_key: ref_point[lon_key],
#                     lat_key: ref_point[lat_key],
#                 }
                
#                 gt_pos = qry["gt_pos"]
#                 for j, k in enumerate(benchmark_top_k):
#                     inds_k = inds[0, :k] % len(ref_image_dataset)
#                     if np.any(np.isin(inds_k, gt_pos)):
#                         recalls[j] += 1
            
#             # Visualizations
#             if visualize_top_k:
#                 vis_kwargs = {
#                     "query_image": image,
#                     "query_ix": i,
#                     "inds": inds,
#                     "ref_image_dataset": ref_image_dataset,
#                     "savedir": savedir,
#                 }
#                 if is_distance_based:
#                     vis_kwargs["gdists"] = gdists
#                     vis_kwargs["min_dists_at_k"] = min_dists_at_k
#                     vis_kwargs["rotexp_thetas"] = rotexp_thetas
#                     vis_kwargs["query_theta"] = theta
#                     vis_kwargs["gt_pos"] = gt_pos
#                 else:
#                     vis_kwargs["gt_pos"] = qry["gt_pos"]
#                 visualize_top_k_retrieved(**vis_kwargs)
            
#             if visualize_vlad and qry_residuals is not None:
#                 vlad_module = getattr(model, "vlad", None)
#                 if vlad_module is not None:
#                     visualize_vlad_clusters(
#                         query_image=image,
#                         query_ix=i,
#                         query_residuals=qry_residuals,
#                         vlad=vlad_module,
#                         savedir=savedir,
#                         query_theta=theta,
#                     )
            
#             if visualize_heatmap:
#                 if dataset_type == "VisLoc":
#                     visualize_similarity_heatmap_visloc(
#                         dists=dists,
#                         inds=inds,
#                         ref_image_dataset=ref_image_dataset,
#                         query_ix=i,
#                         query_theta=theta,
#                         savedir=savedir,
#                     )
#                 elif dataset_type == "GURS":
#                     qry_east, qry_north = qry["lon"], qry["lat"]
#                     visualize_similarity_heatmap_gurs(
#                         dists=dists,
#                         inds=inds,
#                         ref_image_dataset=ref_image_dataset,
#                         query_ix=i,
#                         query_theta=theta,
#                         query_northing=qry_north,
#                         query_easting=qry_east,
#                         savedir=savedir,
#                         rotref_exp=getattr(build_config, "ROTREF_EXP", False),
#                     )
    
#     tracemalloc.stop()
    
#     avg_pipeline_time = np.mean(pipeline_times)
#     avg_db_search_time = np.mean(db_search_times)
#     max_memory_peak = np.max(memory_peaks)
#     max_vram_peak = np.max(vram_peaks)
    
#     # Save predicted trajectory to json file
#     with open(os.path.join(savedir, "predicted_trajectory.json"), "w") as f:
#         json.dump(predicted_trajectory, f, indent=4)
    
#     # Write results
#     print_file = os.path.join(savedir, "results.txt")
#     with open(print_file, "w") as file:
#         resdict_vdb = {
#             "Model": model_type,
#             "Dataset": dataset_type,
#             "Vec. Database size": db.size(),
#             "Vector dimension": db.vdim,
#             "Num. query images": num_qry_images,
#         }
#         write_resdict_to_file(file, resdict_vdb, header="Metadata")
        
#         resdict_perf = {
#             "Avg. pipeline time": f"{avg_pipeline_time:.4f} s",
#             "Avg. db search time": f"{avg_db_search_time:.4f} s",
#             "Max memory peak": f"{max_memory_peak:.4f} GB",
#             "Max vram peak": f"{max_vram_peak:.4f} GB",
#         }
#         write_resdict_to_file(file, resdict_perf, header="Performance")
        
#         if is_distance_based:
#             # Distance-based metrics
#             all_min_dists_at_k = np.array(all_min_dists_at_k)
#             avg_min_dist_at_k = np.mean(all_min_dists_at_k, axis=0)
#             avg_min_dist_data = [
#                 [k, avg_min_dist_at_k[i]] for i, k in enumerate(benchmark_top_k)
#             ]
#             write_pretty_table(
#                 file=file,
#                 field_names=["Top-k", "Avg. min distance (m)"],
#                 data=avg_min_dist_data,
#                 align="l",
#                 header="Avg. Min Distance",
#             )
            
#             if len(benchmark_recall_at_xmeters) > 0:
#                 recalls_at_xmeters = recalls_at_xmeters / num_qry_images
#                 recall_data = [
#                     [xmeters, recalls_at_xmeters[i]]
#                     for i, xmeters in enumerate(benchmark_recall_at_xmeters)
#                 ]
#                 write_pretty_table(
#                     file=file,
#                     field_names=["Distance (m)", "Recall@1"],
#                     data=recall_data,
#                     align="l",
#                     header="Distance Recalls",
#                 )
            
#             intersection_recalls_at_k = intersection_recalls_at_k / num_qry_images
#             intersection_recall_data = [
#                 [k, intersection_recalls_at_k[i]] for i, k in enumerate(benchmark_top_k)
#             ]
#             write_pretty_table(
#                 file=file,
#                 field_names=["Top-k", "Intersection Recall"],
#                 data=intersection_recall_data,
#                 align="l",
#                 header="Intersection Recalls",
#             )
#         else:
#             # Index-based metrics
#             recalls = recalls / num_qry_images
#             recall_data = [[k, recalls[i]] for i, k in enumerate(benchmark_top_k)]
#             write_pretty_table(
#                 file=file,
#                 field_names=["Top-k", "Recall"],
#                 data=recall_data,
#                 align="l",
#                 header="Recall",
#             )
    
#     print(f"Results saved to {print_file}")
#     save_config(traj_config, savedir, prefix="benchmark")

#     return recalls if not is_distance_based else intersection_recalls_at_k


def main():
    parser = create_argparse()
    args = parser.parse_args()
    traj_config = load_config(args.traj_config)
    benchmark_main(args.vdbdir, traj_config)


if __name__ == "__main__":
    main()
