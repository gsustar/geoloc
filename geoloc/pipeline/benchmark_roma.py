"""
Benchmark script for RomaMatchAnythingReranker.
Runs a warmup stage then a profiled benchmark stage using pyinstrument.
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torchvision

# ---------------------------------------------------------------------------
# Paths — edit these or override via CLI args
# ---------------------------------------------------------------------------
DEFAULT_GEOLOC_PATH   = "/home/grega/geoloc"
DEFAULT_HOME_PATH     = "/home/grega"
DEFAULT_CUDA_DEVICE   = "0"
DEFAULT_WEIGHTS       = "/storage/datasets/AerialLoc/Drone2Sat/ckpts/matchanything_weights/matchanything_roma.ckpt"
DEFAULT_GURS_ROOT     = "/storage/private/MORS/gurs"
DEFAULT_AFX_DIR       = "/storage/private/MORS/AFORMX_GOPRO/frames_3fps_1080p/AFormX-flight2-part1"
DEFAULT_AFX_META      = "/storage/private/MORS/AFORMX_GOPRO/frames_3fps_1080p/AFormX-flight2-part1_3fps_1080p_telemetry.csv"
DEFAULT_RESULTS_PATH  = "/storage/datasets/AerialLoc/Drone2Sat/vdbs/contrastivemodel/GURS/secondtry/benchmark4/results_metastats.json"
DEFAULT_QUERY_IX      = 700
DEFAULT_TOP_K         = 100
DEFAULT_BATCH_SIZE    = 5
DEFAULT_IMAGE_SIZE    = 420
DEFAULT_WARMUP_ITERS  = 3



def parse_args():
    p = argparse.ArgumentParser(description="Benchmark RomaMatchAnythingReranker")
    p.add_argument("--cuda-device",   default=DEFAULT_CUDA_DEVICE)
    p.add_argument("--geoloc-path",   default=DEFAULT_GEOLOC_PATH)
    p.add_argument("--home-path",     default=DEFAULT_HOME_PATH)
    p.add_argument("--weights",       default=DEFAULT_WEIGHTS)
    p.add_argument("--gurs-root",     default=DEFAULT_GURS_ROOT)
    p.add_argument("--afx-dir",       default=DEFAULT_AFX_DIR)
    p.add_argument("--afx-meta",      default=DEFAULT_AFX_META)
    p.add_argument("--results-path",  default=DEFAULT_RESULTS_PATH)
    p.add_argument("--query-ix",      type=int, default=DEFAULT_QUERY_IX)
    p.add_argument("--top-k",         type=int, default=DEFAULT_TOP_K)
    p.add_argument("--batch-size",    type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--image-size",    type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--warmup-iters",  type=int, default=DEFAULT_WARMUP_ITERS)
    p.add_argument("--profile-interval", type=float, default=0.004,
                   help="pyinstrument sampling interval in seconds")
    p.add_argument("--output-html",   default="profile_output.html",
                   help="Path to save pyinstrument HTML report")
    return p.parse_args()


def setup_environment(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    for path in (args.geoloc_path, args.home_path):
        if path not in sys.path:
            sys.path.insert(0, path)


def build_datasets(args):
    from geoloc.data.datasets.GURS import GURSReferenceDataset
    from geoloc.data.datasets.ViCoS import AformXQueryImages

    resize = torchvision.transforms.Resize((args.image_size, args.image_size))

    dataset_gurs = GURSReferenceDataset(
        root=args.gurs_root,
        border="Savinjska",
        tile_size=2000,
        stride=1000,
        exclude_slo_border_tifs=True,
        transforms=torchvision.transforms.Compose([resize]),
    )

    dataset_afx = AformXQueryImages(
        query_dir=args.afx_dir,
        metadata_path=args.afx_meta,
        skip_first_n_frames=600,
        skip_last_n_frames=600,
        transforms=torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(1080),
            resize,
        ]),
    )

    return dataset_gurs, dataset_afx


def build_reranker(args, device):
    from geoloc.geoloc.opencv import HomographyEstimator
    from geoloc.models.matchers import RomaMatchAnythingMatcher

    ransac = HomographyEstimator(maxIters=2000, reproj_threshold=1.0)
    reranker = RomaMatchAnythingMatcher(
        coarse_res=args.image_size,
        upsample_res=864,
        upsample_preds=False,
        symmetric=False,
        weights_path=args.weights,
        device=device,
        do_compile=True,
        num_sample_keypoints=5000
    )
    return reranker, ransac


def load_query_data(args, dataset_gurs, dataset_afx, device):
    results = json.load(open(args.results_path))
    ix = args.query_ix

    ret_inds = results[str(ix)]["top100_inds"][: args.top_k]
    reference_images = torch.stack(
        [dataset_gurs[idx]["image"] for idx in ret_inds]
    ).to(device)

    query_image = dataset_afx[ix]["image"].unsqueeze(0).to(device)
    query_image = query_image.expand(reference_images.shape[0], -1, -1, -1)

    inds  = np.array([ret_inds], dtype=np.int64)
    dists = np.zeros((1, len(ret_inds)), dtype=np.float32)

    return query_image, reference_images, inds, dists, results, ix


def run_warmup(reranker, ransac,dataset_gurs, query_image, inds, dists, device, args):
    """Warm up by calling rerank() — the exact same codepath as the benchmark.
    This ensures torch.compile fully traces and caches before profiling starts.
    Using a different query index avoids any caching effects on the real benchmark.
    """
    from geoloc.eval.reranking import rerank

    print(f"\n{'='*60}")
    print(f"  WARMUP  ({args.warmup_iters} iterations via rerank(), "
          f"batch_size={args.batch_size})")
    print(f"  (first iter may be slow — this is torch.compile tracing)")
    print(f"{'='*60}")

    with torch.amp.autocast(dtype=torch.float16, device_type=device.type):
        for i in range(args.warmup_iters):
            t0 = time.perf_counter()
            _ = rerank(
                reranker,
                ransac,
                query_image[0],
                dataset_gurs,
                inds,
                dists,
                device=device,
                batch_size=args.batch_size,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            compiled = " ← compile" if i == 0 else ""
            print(f"  warmup iter {i+1}/{args.warmup_iters}  {elapsed:.2f}s{compiled}")

    print("  Warmup complete.\n")


def run_benchmark(reranker, ransac, dataset_gurs, query_image, inds, dists, device,
                  args, results, ix):
    from geoloc.eval.reranking import rerank
    from pyinstrument import Profiler

    print(f"{'='*60}")
    print(f"  BENCHMARK  (top-{args.top_k}, batch_size={args.batch_size})")
    print(f"  query index: {ix}")
    print(f"{'='*60}\n")

    profiler = Profiler(interval=args.profile_interval)

    torch.cuda.synchronize()
    wall_start = time.perf_counter()

    profiler.start()

    with torch.amp.autocast(dtype=torch.float16, device_type=device.type):
        rerank_res = rerank(
            reranker,
            ransac,
            query_image[0],
            dataset_gurs,
            inds,
            dists,
            device=device,
            batch_size=args.batch_size,
        )

    torch.cuda.synchronize()
    profiler.stop()

    wall_elapsed = time.perf_counter() - wall_start

    # --- Console report ---
    print(profiler.output_text(unicode=True, color=True, timeline=False))

    # --- HTML report ---
    html = profiler.output_html()
    with open(args.output_html, "w") as f:
        f.write(html)
    print(f"\nHTML profile saved → {args.output_html}")

    # --- Summary ---
    all_inliers = np.array(rerank_res["all_num_inliers"])
    best_ix     = int(np.argmax(all_inliers))
    print(f"\n{'='*60}")
    print(f"  Wall time : {wall_elapsed:.3f}s")
    print(f"  Pairs     : {args.top_k}")
    print(f"  Per pair  : {wall_elapsed / args.top_k * 1000:.1f}ms")
    print(f"  Best match: index {best_ix}  ({all_inliers[best_ix]} inliers)")
    print(f"  GT rank1  : {results[str(ix)]['rank1']}")
    print(f"{'='*60}\n")

    return rerank_res

def plot_inlier_distribution(rerank_res):
    import matplotlib.pyplot as plt

    all_inliers = np.array(rerank_res["all_num_inliers"])
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(all_inliers)), all_inliers)
    plt.xlabel("Reference Image Index")
    plt.ylabel("Number of Inliers")
    plt.title("Inlier Distribution Across Reference Images")
    plt.xticks(range(len(all_inliers)))
    plt.grid(axis="y")
    plt.savefig("inlier_distribution.png")
    

def main():
    args = parse_args()
    setup_environment(args)

    # cuDNN autotuner — helps for fixed input sizes
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading datasets...")
    dataset_gurs, dataset_afx = build_datasets(args)

    print("Building reranker...")
    reranker, ransac = build_reranker(args, device)
    reranker = reranker.to(device)


    # test if the model exports cleanly
    dummy_A = torch.randn(DEFAULT_BATCH_SIZE, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, device=device)
    dummy_B = torch.randn(DEFAULT_BATCH_SIZE, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, device=device)

    print("Loading query data...")
    query_image, reference_images, inds, dists, results, ix = load_query_data(
        args, dataset_gurs, dataset_afx, device
    )
    print(f"  query shape    : {query_image.shape}")
    print(f"  reference shape: {reference_images.shape}")

    # -----------------------------------------------------------------------
    run_warmup(reranker, ransac, dataset_gurs, query_image, inds, dists, device, args)
    # -----------------------------------------------------------------------
    rerank_res = run_benchmark(reranker, ransac, dataset_gurs, query_image, inds, dists,
                  device, args, results, ix)
    plot_inlier_distribution(rerank_res)


if __name__ == "__main__":
    main()