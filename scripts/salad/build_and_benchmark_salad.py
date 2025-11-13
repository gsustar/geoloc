# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/salad/build_and_benchmark_salad.py --dataset VPAir

import os
import argparse

from geoloc.config_parser import load_config
from geoloc.utils import color_text
from geoloc.data.datasets.VisLoc import VISLOC_TEST_ROUNDS

from build_vdb_salad import build_vdb_salad
from benchmark_salad_VPAir import benchmark_salad_vpair
from benchmark_salad_VisLoc import benchmark_salad_visloc

# from benchmark_salad_GURS import benchmark_salad_gurs
# from benchmark_salad_ALTO import benchmark_salad_alto


def create_argparse():
    parser = argparse.ArgumentParser(description="SALAD Benchmarking")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset to benchmark on"
    )
    return parser


def main():
    parser = create_argparse()
    args = parser.parse_args()

    build_vdb_config_path = (
        f"configs/salad/{args.dataset}/build_vdb_salad_{args.dataset}.yaml"
    )
    benchmark_config_path = (
        f"configs/salad/{args.dataset}/benchmark_salad_{args.dataset}.yaml"
    )
    build_vdb_config = load_config(build_vdb_config_path)
    benchmark_config = load_config(benchmark_config_path)

    base_savedir = build_vdb_config.savedir
    if args.dataset == "VPAir":
        build_vdb_salad(build_vdb_config)
        benchmark_salad_vpair(base_savedir, benchmark_config)

    elif args.dataset == "VisLoc":
        for flight_idx in VISLOC_TEST_ROUNDS:
            print(f"Processing flight {flight_idx}...")
            build_vdb_config.dataset.init_args.flight_idx = flight_idx
            benchmark_config.dataset.init_args.flight_idx = flight_idx
            # new_savedir = os.path.join(base_savedir, str(flight_idx).zfill(2))
            new_savedir = os.path.join(base_savedir, flight_idx)
            build_vdb_config.savedir = new_savedir
            try:
                build_vdb_salad(build_vdb_config)
                benchmark_salad_visloc(new_savedir, benchmark_config)
            except Exception as e:
                print(color_text(f"Error processing flight {flight_idx}: {e}", "red"))
                continue

    elif args.dataset == "GURS":
        # TODO: add trajectories for GURS
        build_vdb_salad(build_vdb_config)
        benchmark_salad_gurs(base_savedir, benchmark_config)

    elif args.dataset == "ALTO":
        rounds = [1]
        for _round in rounds:
            print(f"Processing round {_round}...")
            build_vdb_config.dataset.init_args.round = _round
            benchmark_config.dataset.init_args.round = _round
            new_savedir = os.path.join(
                base_savedir, str(_round), build_vdb_config.dataset.init_args.split
            )
            build_vdb_config.savedir = new_savedir
            try:
                build_vdb_salad(build_vdb_config)
                benchmark_salad_alto(new_savedir, benchmark_config)
            except Exception as e:
                print(color_text(f"Error processing round {_round}: {e}", "red"))
                continue
    else:
        raise ValueError("Unsupported dataset")


if __name__ == "__main__":
    main()
