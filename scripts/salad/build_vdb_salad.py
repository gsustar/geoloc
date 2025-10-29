# CUDA_VISIBLE_DEVICES=1 DEBUG=0 PYTHONPATH=. python scripts/salad/build_vdb_salad.py --config configs/salad/VPAir/build_vdb_salad_VPAir.yaml --batchsize 16

import os
import torch
import argparse

from torch.utils.data import DataLoader

from geoloc.config_parser import load_config, save_config, class_from_config
from geoloc.utils import DEBUG
from geoloc.data.utils import collate_with_geometry


def create_argparse():
	parser = argparse.ArgumentParser(description="SALAD Vector Database Builder")
	parser.add_argument("--config", type=str, help="Path to the configuration file")
	parser.add_argument("--batchsize", type=int, default=16, help="Batch size for processing images")
	return parser

@torch.no_grad()
def build_vdb_salad(config, batchsize=16):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	savedir = config.savedir
	os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)

	ref_image_dataset = class_from_config(config.dataset)
	ref_image_dataloader = DataLoader(ref_image_dataset, batch_size=batchsize, shuffle=False, collate_fn=collate_with_geometry)

	salad_cls, _ = class_from_config(config.salad, instantiate=False)
	salad = salad_cls.load_from_checkpoint(config.salad.checkpoint, strict=True).to(device)
	salad.eval()

	print("Building database...")
	rotexp_thetas = [0, 90, 180, 270] if config.ROTREF_EXP else [0]
	db = class_from_config(config.vdb)
	db.build(
		ref_image_dataloader, 
		model=salad, 
		rotation_angles=rotexp_thetas, 
		device=device)

	print("Saving database...")
	db.save(savedir)
	save_config(config, savedir)


def main():
	parser = create_argparse()
	args = parser.parse_args()
	config = load_config(args.config)
	build_vdb_salad(config, batchsize=args.batchsize)


if __name__ == "__main__":
	main()