# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/anyloc/build_benchmark_anyloc.py --dataset VPAir

import os
import argparse

from geoloc.config_parser import load_config
from geoloc.utils import color_text

from geoloc.geoloc.pipeline.train import train
from build import build_vdb
from benchmark import benchmark

def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=False, help="Model to use for the pipeline"
	)
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Dataset to use for the pipeline"
    )
    return parser


def pipeline(model_name, dataset_name):
    train_config_path = f"configs/{model_name}/{dataset_name}/train_config.yaml"
    build_vdb_config_path = (
        f"configs/{model_name}/{dataset_name}/build_config.yaml"
    )
    benchmark_config_path = (
        f"configs/{model_name}/{dataset_name}/benchmark_config.yaml"
    )
    train_config = load_config(train_config_path)
    build_config = load_config(build_vdb_config_path)
    benchmark_config = load_config(benchmark_config_path)

    base_savedir = train_config.savedir
    if dataset_name == "VPAir":
        train(train_config)
        build_vdb(build_config)
        benchmark(base_savedir, benchmark_config)

    elif dataset_name == "VisLoc":
        flight_idxs = [i for i in range(1, 12) if i != 7]
        for flight_idx in flight_idxs:
            print(f"Processing flight {flight_idx}...")
            build_config.dataset.init_args.flight_idx = flight_idx
            benchmark_config.dataset.init_args.flight_idx = flight_idx
            new_savedir = os.path.join(base_savedir, str(flight_idx).zfill(2))
            build_config.savedir = new_savedir
            # Ensure VLAD cache_dir is set to the current savedir
            build_config.anyloc.init_args.vlad.init_args.cache_dir = new_savedir
            try:
                train(train_config)
                build_vdb(build_config)
                benchmark(new_savedir, benchmark_config)
            except Exception as e:
                print(color_text(f"Error processing flight {flight_idx}: {e}", "red"))
                continue

    elif dataset_name == "GURS":
        # TODO: add trajectories for GURS
        train(train_config)
        build_vdb(build_config)
        benchmark(base_savedir, benchmark_config)

    elif dataset_name == "ALTO":
        rounds = [1]
        for _round in rounds:
            print(f"Processing round {_round}...")
            build_config.dataset.init_args.round = _round
            benchmark_config.dataset.init_args.round = _round
            new_savedir = os.path.join(
                base_savedir, str(_round), build_config.dataset.init_args.split
            )
            build_config.savedir = new_savedir
            # Ensure VLAD cache_dir is set to the current savedir
            build_config.anyloc.init_args.vlad.init_args.cache_dir = new_savedir
            try:
                train(train_config)
                build_vdb(build_config)
                benchmark(new_savedir, benchmark_config)
            except Exception as e:
                print(color_text(f"Error processing round {_round}: {e}", "red"))
                continue
    else:
        raise ValueError("Unsupported dataset")


def main():
    parser = create_argparse()
    args = parser.parse_args()
    pipeline(args.model_name, args.dataset_name)


if __name__ == "__main__":
    main()
