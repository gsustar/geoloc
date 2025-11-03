# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/anyloc/build_benchmark_anyloc.py --dataset VPAir

import os
import argparse

from geoloc.config_parser import load_config
from geoloc.utils import color_text
from build_vdb_anyloc import build_vdb_anyloc
from benchmark_anyloc_VPAir import benchmark_anyloc_vpair
from benchmark_anyloc_VisLoc import benchmark_anyloc_visloc
from benchmark_anyloc_GURS import benchmark_anyloc_gurs
from benchmark_anyloc_ALTO import benchmark_anyloc_alto


def create_argparse():
	parser = argparse.ArgumentParser(description="AnyLoc Benchmarking")
	parser.add_argument("--dataset", type=str, required=True, help="Dataset to benchmark on")
	return parser


def main():
	parser = create_argparse()
	args = parser.parse_args()

	build_vdb_config_path = f"configs/anyloc/{args.dataset}/build_vdb_anyloc_{args.dataset}.yaml"
	benchmark_config_path = f"configs/anyloc/{args.dataset}/benchmark_anyloc_{args.dataset}.yaml"
	build_vdb_config = load_config(build_vdb_config_path)
	benchmark_config = load_config(benchmark_config_path)

	base_savedir = build_vdb_config.savedir
	if args.dataset == "VPAir":
		build_vdb_anyloc(build_vdb_config)
		benchmark_anyloc_vpair(base_savedir, benchmark_config)

	elif args.dataset == "VisLoc":
		flight_idxs = [i for i in range(1,12) if i != 7]
		for flight_idx in flight_idxs:
			print(f"Processing flight {flight_idx}...")
			build_vdb_config.dataset.init_args.flight_idx = flight_idx
			benchmark_config.dataset.init_args.flight_idx = flight_idx
			new_savedir = os.path.join(base_savedir, str(flight_idx).zfill(2))
			build_vdb_config.savedir = new_savedir
			# Ensure VLAD cache_dir is set to the current savedir
			build_vdb_config.anyloc.init_args.vlad.init_args.cache_dir = new_savedir
			try:
				build_vdb_anyloc(build_vdb_config)
				benchmark_anyloc_visloc(new_savedir, benchmark_config)
			except Exception as e:
				print(color_text(f"Error processing flight {flight_idx}: {e}", "red"))
				continue

	elif args.dataset == "GURS":
		# TODO: add trajectories for GURS
		build_vdb_anyloc(build_vdb_config)
		benchmark_anyloc_gurs(base_savedir, benchmark_config)

	elif args.dataset == "ALTO":
		rounds = [1, 2]
		for _round in rounds:
			print(f"Processing round {_round}...")
			build_vdb_config.dataset.init_args.round = _round
			benchmark_config.dataset.init_args.round = _round
			new_savedir = os.path.join(
				base_savedir, str(_round), build_vdb_config.dataset.init_args.split)
			build_vdb_config.savedir = new_savedir
			# Ensure VLAD cache_dir is set to the current savedir
			build_vdb_config.anyloc.init_args.vlad.init_args.cache_dir = new_savedir
			try:
				build_vdb_anyloc(build_vdb_config)
				benchmark_anyloc_alto(new_savedir, benchmark_config)
			except Exception as e:
				print(color_text(f"Error processing round {_round}: {e}", "red"))
				continue
	else:
		raise ValueError("Unsupported dataset")


if __name__ == "__main__":
	main()