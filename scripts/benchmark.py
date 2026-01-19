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
from geoloc.utils import DEBUG, load_model
from geoloc.eval.metrics import calculate_distances, calculate_intersections
from geoloc.eval.utils import write_resdict_to_file, write_pretty_table

INDEX_BASED_DATASETS = ["vpair", "alto"]
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
    if dataset_type in ["ALTO", "GURS"]:
        return ("east", "north")
    return ("lon", "lat")

def benchmark_main(vdbdir: str, traj_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_config = load_config(os.path.join(vdbdir, "build_config.yaml"))
    model = load_model(build_config).eval().to(device)

    ref_image_dataset = class_from_config(build_config.dataset)
    qry_image_dataset = class_from_config(traj_config.dataset)

    model_type = model.__class__.__name__
    dataset_type = traj_config.dataset.class_path.split(".")[-2]
    print(f"Detected model: {model_type}")
    print(f"Detected dataset: {dataset_type}")

    savedir = get_savedir(build_config.savedir)
    os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)

    visualize_vlad = getattr(traj_config, "VISUALIZE_VLAD", False) and supports_vlad_visualization(model_type)
    visualize_top_k = getattr(traj_config, "VISUALIZE_TOP_K", False)
    visualize_heatmap = getattr(traj_config, "VISUALIZE_HEATMAP", False)
    rotexp_thetas = [0, 90, 180, 270] if getattr(build_config, "ROTREF_EXP", False) else [0]
    rottraj_thetas = [0, 90, 180, 270] if getattr(traj_config, "ROTTRAJ_EXP", False) else [0]
    benchmark_recall_at_xmeters = getattr(traj_config, "benchmark_recall_at_xmeters", [])
    benchmark_top_k = traj_config.benchmark_top_k
    every_n = traj_config.every_n

    if getattr(traj_config, "ROTTRAJ_EXP", False):
        print("Running rotation trajectory experiment...")

    print(f"Loading Vector Database for {model_type}...")
    vdb = load_database(vdbdir, build_config)

    benchmark_results = benchmark(
        model=model, model_type=model_type, vdb=vdb, qry_image_dataset=qry_image_dataset,
        ref_image_dataset=ref_image_dataset, dataset_type=dataset_type,
        benchmark_recall_at_xmeters=benchmark_recall_at_xmeters, benchmark_top_k=benchmark_top_k,
        rotexp_thetas=rotexp_thetas, rottraj_thetas=rottraj_thetas, visualize_vlad=visualize_vlad, 
        visualize_top_k=visualize_top_k, visualize_heatmap=visualize_heatmap, every_n=every_n,
        savedir=savedir, device=device
    )

    num_qry_images = benchmark_results["num_qry_images"]
    predicted_trajectory = benchmark_results["predicted_trajectory"]
    avg_pipeline_time = benchmark_results["avg_pipeline_time"]
    avg_db_search_time = benchmark_results["avg_db_search_time"]
    max_memory_peak = benchmark_results["max_memory_peak"]
    max_vram_peak = benchmark_results["max_vram_peak"]
    is_distance_based = benchmark_results["is_distance_based"]

    # Save predicted trajectory to json file
    with open(os.path.join(savedir, "predicted_trajectory.json"), "w") as f:
        json.dump(predicted_trajectory, f, indent=4)
    
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
                header="Avg. Min Distance",
            )
            
            if len(benchmark_recall_at_xmeters) > 0:
                recalls_at_xmeters = recalls_at_xmeters / num_qry_images
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
            
            intersection_recalls_at_k = intersection_recalls_at_k / num_qry_images
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
            recalls = recalls / num_qry_images
            recall_data = [[k, recalls[i]] for i, k in enumerate(benchmark_top_k)]
            write_pretty_table(
                file=file,
                field_names=["Top-k", "Recall"],
                data=recall_data,
                align="l",
                header="Recall",
            )
    print(f"Results saved to {print_file}")
    save_config(traj_config, savedir, prefix="benchmark")


@torch.no_grad()
def benchmark(
    model, model_type, vdb, qry_image_dataset, ref_image_dataset, 
    dataset_type, benchmark_recall_at_xmeters=[100, 250, 500, 1000], 
    benchmark_top_k=[1, 5, 10, 25, 50, 100], rotexp_thetas=[0], rottraj_thetas=[0], 
    visualize_vlad=False, visualize_top_k=False, visualize_heatmap=False, every_n=1, 
    savedir=None, device="cpu"
):
    # Distance-based metrics (for VisLoc, GURS)
    is_distance_based = dataset_type in DISTANCE_BASED_DATASETS
    lon_key, lat_key = get_position_keys(dataset_type)
    
    print(f"Benchmarking {model_type} on {dataset_type}...")
    pipeline_times = []
    db_search_times = []
    memory_peaks = []
    vram_peaks = []
    
    tracemalloc.start()
    
    if is_distance_based:
        all_min_dists_at_k = []
        recalls_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
        intersection_recalls_at_k = np.zeros(len(benchmark_top_k))
    else:
        recalls = np.zeros(len(benchmark_top_k))
    
    num_qry_images = 0
    predicted_trajectory = {}
    for j, theta in enumerate(rottraj_thetas):
        for i, qry in enumerate(tqdm(qry_image_dataset)):
            if DEBUG > 1 and i > 10:
                break
            if i % every_n != 0:
                continue
            num_qry_images += 1
            image = qry["image"].to(device)
            
            if theta != 0:
                image = TF.rotate(image, theta)
            
            model_args = {}
            if visualize_vlad:
                model_args["return_residuals"] = True
            if model_type in ["SegVLAD", "Mast3rRetrievalModel"]:
                model_args["idx"] = i + j * len(qry_image_dataset)

            pipeline_start_time = time.time()
            outdict = model(image.unsqueeze(0), **model_args)
            pipeline_times.append(time.time() - pipeline_start_time)

            x = outdict["out"]
            qry_residuals = outdict.get("qry_residuals", None)
            
            search_top_k = [vdb.size()] if visualize_heatmap else benchmark_top_k

            search_args = {}
            if model_type in ["SegVLAD", "Mast3rRetrievalModel"]:
                search_args["imids"] = outdict["ids"].cpu().numpy()
            db_search_start_time = time.time()
            dists, inds = vdb.search(qu=x, k=max(search_top_k), **search_args)
            db_search_times.append(time.time() - db_search_start_time)
            
            curr_iter_memory_peak = bytes_to_gb(tracemalloc.get_traced_memory()[1])
            curr_iter_vram_peak = bytes_to_gb(torch.cuda.max_memory_allocated())
            
            memory_peaks.append(curr_iter_memory_peak)
            vram_peaks.append(curr_iter_vram_peak)
            
            torch.cuda.reset_peak_memory_stats()
            tracemalloc.reset_peak()
            
            if is_distance_based:
                gdists, ref_points = calculate_distances(
                    qry, benchmark_top_k, inds, qry_image_dataset, ref_image_dataset
                )
                predicted_trajectory[i] = {
                    lon_key: ref_points[0][0],
                    lat_key: ref_points[0][1],
                }
                
                # Calculate top-k distances
                min_dists_at_k = np.zeros(len(benchmark_top_k))
                for j, k in enumerate(benchmark_top_k):
                    min_dists_at_k[j] = np.min(gdists[:k])
                all_min_dists_at_k.append(min_dists_at_k)
                
                # Calculate intersections
                gt_pos, intersection_tps = calculate_intersections(
                    qry, benchmark_top_k, inds, ref_image_dataset
                )
                
                for j, k in enumerate(benchmark_top_k):
                    if np.any(intersection_tps[:k]):
                        intersection_recalls_at_k[j] += 1
                
                # Recall@x meters
                for j, xmeters in enumerate(benchmark_recall_at_xmeters):
                    if min_dists_at_k[0] <= xmeters:
                        recalls_at_xmeters[j] += 1
            else:
                ref_point = ref_image_dataset[(inds[0, 0] % len(ref_image_dataset)).item()]
                predicted_trajectory[i] = {
                    lon_key: ref_point[lon_key],
                    lat_key: ref_point[lat_key],
                }
                
                gt_pos = qry["gt_pos"]
                for j, k in enumerate(benchmark_top_k):
                    inds_k = inds[0, :k] % len(ref_image_dataset)
                    if np.any(np.isin(inds_k, gt_pos)):
                        recalls[j] += 1
            
            # Visualizations
            if visualize_top_k:
                vis_kwargs = {
                    "query_image": image,
                    "query_ix": i,
                    "inds": inds,
                    "ref_image_dataset": ref_image_dataset,
                    "savedir": savedir,
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
                        query_ix=i,
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
                        query_ix=i,
                        query_theta=theta,
                        savedir=savedir,
                    )
                elif dataset_type == "GURS":
                    qry_east, qry_north = qry["lon"], qry["lat"]
                    visualize_similarity_heatmap_gurs(
                        dists=dists,
                        inds=inds,
                        ref_image_dataset=ref_image_dataset,
                        query_ix=i,
                        query_theta=theta,
                        query_northing=qry_north,
                        query_easting=qry_east,
                        savedir=savedir,
                        # rotref_exp=getattr(build_config, "ROTREF_EXP", False),
                        rotref_exp=True if rotexp_thetas != [0] else False,
                    )
    tracemalloc.stop()
    avg_pipeline_time = np.mean(pipeline_times)
    avg_db_search_time = np.mean(db_search_times)
    max_memory_peak = np.max(memory_peaks)
    max_vram_peak = np.max(vram_peaks)

    return dict(
        num_qry_images=num_qry_images,
        predicted_trajectory=predicted_trajectory,
        avg_pipeline_time=avg_pipeline_time,
        avg_db_search_time=avg_db_search_time,
        max_memory_peak=max_memory_peak,
        max_vram_peak=max_vram_peak,
        recalls=recalls if not is_distance_based else intersection_recalls_at_k,
        recalls_at_xmeters=recalls_at_xmeters if is_distance_based else None,
        all_min_dists_at_k=all_min_dists_at_k if is_distance_based else None,
        is_distance_based=is_distance_based,
    )


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
