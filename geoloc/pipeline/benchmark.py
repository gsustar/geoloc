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
    visualize_matches2,
    visualize_warp,
)
from geoloc.data.vdb import load_database
from geoloc.config_parser import load_config, save_config, class_from_config
from geoloc.utils import DEBUG, crs_transform, load_model, get_model_type, get_dataset_type, requires_arg
from geoloc.eval.metrics import calculate_distances, calculate_intersections, hit_at_k, tp_at_k, safe_rank1, rank1_or_sentinel, reciprocal_rank_from_rank1, average_precision
from geoloc.eval.utils import write_resdict_to_file, write_pretty_table, segvlad_get_matches, inlier_distribution_check, get_ref_points
from geoloc.eval.homography import predict_qry_camera_position
from geoloc.data.utils import collate_with_geometry
from geoloc.models.reranking import rerank

INDEX_BASED_DATASETS = ["vpair", "alto", "ortholoc"]
DISTANCE_BASED_DATASETS = ["visloc", "ges", "vicos"]

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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
    parser.add_argument(
        "--disable_fp16", action="store_true", help="Disable FP16 precision during benchmarking"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable pyinstrument profiling for benchmark loop"
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
    if dataset_type in ["alto", "vicos", "ges"]:
        return ("east", "north")
    return ("lon", "lat")


def _build_spatial_coords(ref_image_dataset, lon_key: str, lat_key: str, target_size: int):
    def _coords_from_all_windows(all_windows):
        bounds = all_windows.geometry.bounds
        center_x = ((bounds["minx"] + bounds["maxx"]) / 2.0).to_numpy(dtype=np.float64)
        center_y = ((bounds["miny"] + bounds["maxy"]) / 2.0).to_numpy(dtype=np.float64)
        return np.stack([center_y, center_x], axis=1)

    if hasattr(ref_image_dataset, "all_windows"):
        coords = _coords_from_all_windows(ref_image_dataset.all_windows)
    elif hasattr(ref_image_dataset, "_datasets"):
        all_coords = []
        for ds in ref_image_dataset._datasets:
            if not hasattr(ds, "all_windows"):
                raise ValueError("Expected GURS sub-dataset with `all_windows` attribute.")
            all_coords.append(_coords_from_all_windows(ds.all_windows))
        coords = np.concatenate(all_coords, axis=0) if len(all_coords) > 0 else np.empty((0, 2), dtype=np.float64)
    else:
        raise ValueError("Expected GURS reference dataset exposing `all_windows` (or `_datasets` with `all_windows`).")

    if coords.shape[0] == target_size:
        return coords

    if coords.shape[0] == 0 or target_size % coords.shape[0] != 0:
        raise ValueError(
            f"Cannot align spatial coordinates length ({coords.shape[0]}) to vdb size ({target_size})."
        )

    return np.repeat(coords, target_size // coords.shape[0], axis=0)

def benchmark_main(vdbdir: str, traj_config, use_fp16=True, profile=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_config = load_config(os.path.join(vdbdir, "build_config.yaml"))
    model = load_model(build_config).eval().to(device)

    ref_image_dataset = class_from_config(build_config.dataset)
    qry_image_dataset = class_from_config(traj_config.dataset)
    qry_image_dataloader = torch.utils.data.DataLoader(qry_image_dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_with_geometry)

    model_type = get_model_type(model)
    dataset_type = get_dataset_type(qry_image_dataset)
    print(f"Detected model: {model_type}")
    print(f"Detected dataset: {dataset_type}")

    # Distance-based metrics (for VisLoc, Vicos, GES)
    is_distance_based = dataset_type in DISTANCE_BASED_DATASETS
    lon_key, lat_key = get_position_keys(dataset_type)

    savedir = get_savedir(build_config.savedir)
    os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)

    visualize_vlad = getattr(traj_config, "VISUALIZE_VLAD", False) and supports_vlad_visualization(model_type)
    visualize_top_k = getattr(traj_config, "VISUALIZE_TOP_K", False)
    visualize_heatmap = getattr(traj_config, "VISUALIZE_HEATMAP", False)
    visualize_matcher = getattr(traj_config, "VISUALIZE_MATCHER", False)
    save_salad_matrix = getattr(traj_config, "SAVE_SALAD_MATRIX", False)
    rotexp_thetas = [0, 90, 180, 270] if getattr(build_config, "ROTREF_EXP", False) else [0]
    rottraj_thetas = [0, 90, 180, 270] if getattr(traj_config, "ROTTRAJ_EXP", False) else [0]
    benchmark_recall_at_xmeters = getattr(traj_config, "benchmark_recall_at_xmeters", [])
    benchmark_top_k = traj_config.benchmark_top_k
    every_n = traj_config.every_n
    radius_search_meters = float(getattr(traj_config, "radius_search_meters", 0.0))
    confidence_pass_streak_threshold = int(getattr(traj_config, "confidence_pass_streak_threshold", 2))

    if getattr(traj_config, "ROTTRAJ_EXP", False):
        print("Running rotation trajectory experiment...")

    print(f"Loading Vector Database for {model_type}...")
    vdb = load_database(vdbdir, build_config)


    matcher = None
    ransac = None
    rerank_batch_size = 1
    if hasattr(traj_config, "reranking"):
        matcher = class_from_config(traj_config.reranking.matcher)
        ransac = class_from_config(traj_config.reranking.ransac)
        rerank_batch_size = getattr(traj_config.reranking, "batch_size", 1)
    
    use_radius_policy = (
        dataset_type in ["vicos", "ges"] and
        matcher is not None and
        hasattr(vdb, "search_radius")
        and radius_search_meters > 0.0
    )

    if matcher.do_compile:
        print("Forcing compilation of matcher...")
        dummy_input = torch.randn(rerank_batch_size, 3, matcher.coarse_res, matcher.coarse_res).to(device=device, dtype=torch.float16).contiguous()
        for _ in range(10):
            matcher(dummy_input, dummy_input)

    if use_radius_policy:
        try:
            spatial_coords = _build_spatial_coords(ref_image_dataset, lon_key=lon_key, lat_key=lat_key, target_size=vdb.size())
            vdb.set_spatial_index(spatial_coords, crs=ref_image_dataset.crs)
            print(f"Radius search policy enabled for {dataset_type} with radius {radius_search_meters:.1f} m.")
        except Exception as e:
            print(f"Radius search policy disabled: failed to initialize spatial index ({e}).")
            use_radius_policy = False

    loop_kwargs = dict(
        model=model, model_type=model_type, vdb=vdb, qry_image_dataloader=qry_image_dataloader, qry_image_dataset=qry_image_dataset,
        ref_image_dataset=ref_image_dataset, vdbdir=vdbdir, dataset_type=dataset_type, is_distance_based=is_distance_based,
        lon_key=lon_key, lat_key=lat_key, benchmark_recall_at_xmeters=benchmark_recall_at_xmeters,
        benchmark_top_k=benchmark_top_k, rotexp_thetas=rotexp_thetas, rottraj_thetas=rottraj_thetas,
        visualize_vlad=visualize_vlad, visualize_top_k=visualize_top_k, visualize_heatmap=visualize_heatmap, visualize_matcher=visualize_matcher,
        every_n=every_n, savedir=savedir, device=device, save_salad_matrix=save_salad_matrix, use_fp16=use_fp16,
        matcher=matcher, ransac=ransac, rerank_batch_size=rerank_batch_size,
        use_radius_policy=use_radius_policy, radius_search_meters=radius_search_meters, confidence_pass_streak_threshold=confidence_pass_streak_threshold,
    )

    if profile:
        try:
            from pyinstrument import Profiler
        except ImportError as e:
            raise ImportError(
                "Profiling requested, but pyinstrument is not installed. Install it with `pip install pyinstrument` or disable profiling."
            ) from e

        print("pyinstrument profiling enabled.")
        profiler = Profiler()
        profiler.start()
        benchmark_results = benchmark_loop(**loop_kwargs)
        profiler.stop()

        profile_txt_path = os.path.join(savedir, "profile_pyinstrument.txt")
        with open(profile_txt_path, "w") as f:
            f.write(profiler.output_text(unicode=True, color=False))

        profile_html_path = os.path.join(savedir, "profile_pyinstrument.html")
        with open(profile_html_path, "w") as f:
            f.write(profiler.output_html())

        print(f"pyinstrument profile saved to {profile_txt_path} and {profile_html_path}")
    else:
        benchmark_results = benchmark_loop(**loop_kwargs)

    num_qry_images = benchmark_results["num_qry_images"]
    results_metastats = benchmark_results["results_metastats"]
    avg_pipeline_time = benchmark_results["avg_pipeline_time"]
    avg_db_search_time = benchmark_results["avg_db_search_time"]
    max_memory_peak = benchmark_results["max_memory_peak"]
    max_vram_peak = benchmark_results["max_vram_peak"]
    recalls_at_k = benchmark_results["recalls_at_k"]
    recalls_at_xmeters = benchmark_results["recalls_at_xmeters"]
    avg_min_dist_at_k = benchmark_results["avg_min_dists_at_k"]
    mean_reciprocal_rank = benchmark_results["mean_reciprocal_rank"]
    mean_average_precision = benchmark_results["mean_average_precision"]
    
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
    model, model_type, vdb, qry_image_dataloader, qry_image_dataset, ref_image_dataset, 
    dataset_type,  vdbdir, is_distance_based=None, lon_key=None, lat_key=None, 
    benchmark_recall_at_xmeters=[100, 250, 500, 1000], 
    benchmark_top_k=[1, 5, 10, 25, 50, 100], rotexp_thetas=[0], rottraj_thetas=[0], 
    visualize_vlad=False, visualize_top_k=False, visualize_matcher=False, visualize_heatmap=False, every_n=1, 
    savedir=None, device="cpu", save_salad_matrix=False, use_fp16=True, matcher=None,
    ransac=None, rerank_batch_size=1,
    use_radius_policy=False, radius_search_meters=5000.0, confidence_pass_streak_threshold=2,
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
    recalls_at_k = None
    avg_min_dist_at_k = None
    rotator_thetas = []

    all_top100_inds = []
    all_tps_at_k = []
    all_hits_at_k = []
    all_rank1 = []
    all_rank1_sortable = []
    all_reciprocal_rank = []
    all_average_precision = []

    if is_distance_based:
        all_min_dists_at_k = []
        recalls_at_xmeters = np.zeros(len(benchmark_recall_at_xmeters))
    recalls_at_k = np.zeros(len(benchmark_top_k))
    
    num_qry_images = 0
    results_metastats = {}
    confidence_pass_streak = 0
    prev_predicted_coordinates = None

    tracemalloc.start()
    for j, theta in enumerate(rottraj_thetas):
        # for i, qry in enumerate(tqdm(qry_image_dataset)):
        for i, qry in enumerate(tqdm(qry_image_dataloader)):
            if DEBUG > 1 and i > 100:
                break
            if i % every_n != 0:
                continue
            num_qry_images += 1

            use_radius_now = (
                use_radius_policy
                and confidence_pass_streak >= confidence_pass_streak_threshold
                and prev_predicted_coordinates is not None
            )

            benchmark_results = benchmark_single(
                model=model, model_type=model_type, vdb=vdb, qry=qry, qry_image_dataset=qry_image_dataset,
                ref_image_dataset=ref_image_dataset, vdbdir=vdbdir, dataset_type=dataset_type, query_ix=i,
                is_distance_based=is_distance_based, lon_key=lon_key, lat_key=lat_key, theta=theta, theta_ix=j,
                benchmark_recall_at_xmeters=benchmark_recall_at_xmeters, benchmark_top_k=benchmark_top_k,
                rotexp_thetas=rotexp_thetas, visualize_vlad=visualize_vlad, visualize_matcher=visualize_matcher, visualize_top_k=visualize_top_k,
                visualize_heatmap=visualize_heatmap, savedir=savedir, device=device, save_salad_matrix=save_salad_matrix,
                use_fp16=use_fp16, matcher=matcher, ransac=ransac, rerank_batch_size=rerank_batch_size,
                use_radius_search=use_radius_now, radius_center=prev_predicted_coordinates, radius_meters=radius_search_meters,
            )

            if benchmark_results["passed_distribution_check"]:
                confidence_pass_streak += 1
            else:
                confidence_pass_streak = 0
            prev_predicted_coordinates = benchmark_results["predicted_coordinates"]

            if is_distance_based:
                all_min_dists_at_k.append(benchmark_results["min_dists_at_k"])
                recalls_at_xmeters += benchmark_results["recall_at_xmeters"]
            recalls_at_k += benchmark_results["recall_at_k"]

            all_top100_inds.append(benchmark_results["top100_inds"])
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
                "min_dists_at_k": benchmark_results["min_dists_at_k"].tolist() if benchmark_results["min_dists_at_k"] is not None else None,
                "pipeline_time": benchmark_results["pipeline_time"],
                "db_search_time": benchmark_results["db_search_time"],
                "rerank_time": benchmark_results["rerank_time"] if matcher is not None else None,
                "curr_iter_memory_peak": benchmark_results["curr_iter_memory_peak"],
                "curr_iter_vram_peak": benchmark_results["curr_iter_vram_peak"],
                "top100_inds": benchmark_results["top100_inds"],
                "all_num_inliers": benchmark_results["all_num_inliers"],
                # "all_homographies": benchmark_results["all_homographies"],
                "gt_pos": benchmark_results["gt_pos"],
                "tps_at_k": benchmark_results["tps_at_k"].tolist(),
                "hits_at_k": benchmark_results["hits_at_k"].tolist(),
                "rank1": benchmark_results["rank1"],
                "rank1_sortable": benchmark_results["rank1_sortable"],
                "reciprocal_rank": benchmark_results["reciprocal_rank"],
                "average_precision": benchmark_results["average_precision"],
                "rotator_theta": rotator_theta,
                "passed_distribution_check": benchmark_results["passed_distribution_check"],
                "use_radius_search": use_radius_now,
                "confidence_pass_streak": confidence_pass_streak,
                "radius_search_meters": radius_search_meters if use_radius_now else None,
                "radius_search_center": prev_predicted_coordinates if use_radius_now else None,
            }

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
        results_metastats=results_metastats,
        avg_pipeline_time=avg_pipeline_time,
        avg_db_search_time=avg_db_search_time,
        max_memory_peak=max_memory_peak,
        max_vram_peak=max_vram_peak,
        recalls_at_k=recalls_at_k,
        recalls_at_xmeters=recalls_at_xmeters,
        avg_min_dists_at_k=avg_min_dist_at_k,
        rotator_thetas=rotator_thetas,
        mean_reciprocal_rank=mean_reciprocal_rank,
        mean_average_precision=mean_ap,
    )

@torch.no_grad()
def benchmark_single(
    model, model_type, vdb, qry_image_dataset, 
    ref_image_dataset, vdbdir, dataset_type, query_ix,
    is_distance_based=None, lon_key=None, lat_key=None, theta=0, theta_ix=0, benchmark_recall_at_xmeters=[100, 250, 500, 1000],
    benchmark_top_k=[1, 5, 10, 25, 50, 100], rotexp_thetas=[0], visualize_vlad=False, visualize_matcher=False,
    visualize_top_k=False, visualize_heatmap=False, savedir=None, device=None, save_salad_matrix=False, use_fp16=True, qry=None,
    matcher=None, ransac=None, rerank_batch_size=1,
    use_radius_search=False, radius_center=None, radius_meters=5000.0,
):
    if is_distance_based is None:
        is_distance_based = dataset_type in DISTANCE_BASED_DATASETS
    if lon_key is None or lat_key is None:
        lon_key, lat_key = get_position_keys(dataset_type)

    if qry is None:
        qry = qry_image_dataset[query_ix]
    image = qry["image"].squeeze(0)
    if device is not None:
        image = image.to(device)

    if theta != 0:
        image = TF.rotate(image, theta)
    
    model_args = {}
    if visualize_vlad:
        model_args["return_residuals"] = True
    if save_salad_matrix:
        model_args["return_salad_matrix"] = True
    if requires_arg(model.forward, "idx"):
        model_args["idx"] = query_ix + theta_ix * len(qry_image_dataset)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
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
    if use_radius_search and model_type not in ["Mast3rRetrievalModel"]:
        center_y = float(radius_center[lat_key])
        center_x = float(radius_center[lon_key])
        dists, inds = vdb.search_radius(
            qu=x,
            k=max(search_top_k),
            query_y=center_y,
            query_x=center_x,
            radius_m=float(radius_meters),
        )
    else:
        dists, inds = vdb.search(qu=x, k=max(search_top_k), **search_args)
    db_search_time = time.time() - db_search_start_time

    if model_type in ["SegVLAD"]:
        if len(vdb.imInd_buffer) > 0: # necessary for validation during training
            ref_imInds = vdb.imInd_buffer
        else:
            ref_imInds = torch.load(os.path.join(vdbdir, "segvlad_imInds.npy")).numpy()
        # pred, pred_score = segvlad_get_matches(dists, inds, ref_imInds, n=max(benchmark_top_k))
        inds, dists = segvlad_get_matches(dists, inds, ref_imInds, n=max(benchmark_top_k)) # !! this is not distances, just accumulated similarities
    
    all_num_inliers = None
    all_homographies = None
    passed_distribution_check = None
    if matcher is not None:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            rerank_start_time = time.time()
            rerank_res = rerank(matcher, ransac, image, ref_image_dataset, inds, dists, device=device, batch_size=rerank_batch_size)
            rerank_time = time.time() - rerank_start_time
        dists = rerank_res["dists"]
        inds = rerank_res["inds"]
        all_num_inliers = rerank_res["all_num_inliers"]
        all_num_outliers = rerank_res["all_num_outliers"]
        all_homographies = rerank_res["all_homographies"]
        passed_distribution_check, idcscore = inlier_distribution_check(all_num_inliers, threshold=0.9, max_keypoints=matcher.num_sample_keypoints)

    curr_iter_memory_peak = bytes_to_gb(tracemalloc.get_traced_memory()[1])
    curr_iter_vram_peak = bytes_to_gb(torch.cuda.max_memory_allocated())
    
    predicted_coordinate = None
    recall_at_k = None
    recall_at_xmeters = None
    min_dists_at_k = None

    if is_distance_based:
        ref_points = get_ref_points(inds, ref_image_dataset, benchmark_top_k)
        # gdists, ref_points = calculate_distances(
        #     qry, benchmark_top_k, inds, qry_image_dataset, ref_image_dataset, lon_key, lat_key
        # )
        # top_ref_point = ref_points[0]
        if (all_homographies is not None 
            and len(all_homographies) > 0
            and dataset_type in ["vicos", "ges"]
        ):
            sample_ref_image = ref_image_dataset[0]["image"]
            ref_h, ref_w = int(sample_ref_image.shape[1]), int(sample_ref_image.shape[2])
            qry_h, qry_w = int(image.shape[1]), int(image.shape[2])
            for rpi, curr_ref_point in enumerate(ref_points[:5]):
                try:
                    tile_size = ref_image_dataset.tile_size
                except AttributeError:
                    tile_size = ref_image_dataset.get_tile_size_for_index(inds[0, 0] % len(ref_image_dataset)) # Necessary for MultiTileSizeDataset
                # top_ref_point = apply_homography_to_ref_point(
                #     top_ref_point,
                #     all_homographies[0],
                #     query_image_shape=(1080, 1080), #!!! hardcoded for now. problem is resizing in dataset that breaks the pix_res assumption.
                #     reference_image_shape=(tile_size, tile_size),
                #     meters_per_pixel=ref_image_dataset.pxl_res,
                # )
                ref_points[rpi] = predict_qry_camera_position(
                    ref_center_point=curr_ref_point,
                    ref_original_shape=(tile_size, tile_size),
                    qry_matcher_shape=(qry_h, qry_w),
                    ref_matcher_shape=(ref_h, ref_w),
                    H=all_homographies[0],
                )

        predicted_coordinate = {
            lon_key: ref_points[0][0],
            lat_key: ref_points[0][1],
        }


        qry_point = qry[lon_key], qry[lat_key]
        qry_point = crs_transform(qry_point, qry_image_dataset.crs, ref_image_dataset.crs)
        gdists = calculate_distances(qry_point, ref_points, ref_image_dataset)

        # Calculate top-k distances
        min_dists_at_k = np.zeros(len(benchmark_top_k))
        for h, k in enumerate(benchmark_top_k):
            min_dists_at_k[h] = np.min(gdists[:k])
        
        # Calculate intersections
        # gt_pos, intersection_tps = calculate_intersections(
        #     qry, benchmark_top_k, inds, ref_image_dataset
        # )
        # tp_flags = np.array(intersection_tps, dtype=bool)

        gt_pos = ref_image_dataset._get_gt_windows(qry["geometry"][0], overlap_threshold=0.25) # !!! it can happen that no image in reference database has an overlap of X% with query. Either overlap reference database or set lower threshold
        gt_pos = list(gt_pos.index)
        retrieved = (inds[0, :] % len(ref_image_dataset)).astype(int)
        gt_set = set(gt_pos if isinstance(gt_pos, (list, tuple, np.ndarray)) else [gt_pos])
        tp_flags = np.array([r in gt_set for r in retrieved], dtype=bool)

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
    else:
        ref_point = ref_image_dataset[(inds[0, 0] % len(ref_image_dataset)).item()]
        predicted_coordinate = {
            lon_key: ref_point[lon_key],
            lat_key: ref_point[lat_key],
        }
        
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
            # vis_kwargs["gdists"] = gdists
            vis_kwargs["min_dists_at_k"] = min_dists_at_k
            vis_kwargs["rotexp_thetas"] = rotexp_thetas
            vis_kwargs["query_theta"] = theta
            vis_kwargs["gdists"] = gdists

        vis_kwargs["all_num_inliers"] = all_num_inliers
        vis_kwargs["all_num_outliers"] = all_num_outliers
        vis_kwargs["gt_pos"] = gt_pos
        vis_kwargs["passed_distribution_check"] = passed_distribution_check
        vis_kwargs["idcscore"] = idcscore
        # else:
        #     vis_kwargs["gt_pos"] = qry["gt_pos"]
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
        if dataset_type == "visloc":
            visualize_similarity_heatmap_visloc(
                dists=dists,
                inds=inds,
                ref_image_dataset=ref_image_dataset,
                query_ix=query_ix,
                query_theta=theta,
                savedir=savedir,
            )
        elif dataset_type in ["ges", "vicos"]:
            qry_east, qry_north = qry[lon_key], qry[lat_key]
            visualize_similarity_heatmap_gurs(
                dists=dists,
                inds=inds,
                ref_image_dataset=ref_image_dataset,
                query_ix=query_ix,
                query_theta=theta,
                query_northing=qry_north,
                query_easting=qry_east,
                savedir=savedir,
                rotref_exp=True if rotexp_thetas != [0] else False,
            )

    if visualize_matcher:
        assert matcher is not None, "Matcher is required for warping visualizations."
        imA = image.cpu()
        imB = ref_image_dataset[inds[0, 0] % len(ref_image_dataset)]["image"].cpu()
        visualize_matches2(
            imgA_tensor=imA,
            imgB_tensor=imB,
            kptsA=torch.as_tensor(rerank_res["qry_kpts"][0]),
            kptsB=torch.as_tensor(rerank_res["ref_kpts"][0]),
            mask=torch.as_tensor(rerank_res["inliers"][0]),
            always_draw=True,
            save_path=os.path.join(savedir, f"matches_q{query_ix:05d}.png")
        )
        visualize_warp(
            im_A=imA,
            im_B=imB,
            H=torch.as_tensor(rerank_res["all_homographies"][0]),
            save_path=os.path.join(savedir, f"warp_q{query_ix:05d}.png")
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
        rerank_time=rerank_time if matcher is not None else None,
        curr_iter_memory_peak=curr_iter_memory_peak,
        curr_iter_vram_peak=curr_iter_vram_peak,
        predicted_coordinates=predicted_coordinate,
        recall_at_k=recall_at_k,
        recall_at_xmeters=recall_at_xmeters,
        min_dists_at_k=min_dists_at_k,
        rotator_theta=x_rotator_theta,
        top100_inds=inds[0, :100].tolist(), # Save index of top-100 retrieved images
        all_num_inliers=all_num_inliers,
        all_homographies=all_homographies,
        gt_pos=gt_pos,
        tps_at_k=tps_at_k,
        hits_at_k=hits_at_k,
        rank1=rank1,
        rank1_sortable=rank1_sortable,
        reciprocal_rank=rr,
        average_precision=ap,
        passed_distribution_check=passed_distribution_check,
    )


def main():
    parser = create_argparse()
    args = parser.parse_args()
    traj_config = load_config(args.traj_config)
    benchmark_main(args.vdbdir, traj_config, use_fp16=not args.disable_fp16, profile=args.profile)


if __name__ == "__main__":
    main()
