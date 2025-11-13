# CUDA_VISIBLE_DEVICES=1 DEBUG=0 PYTHONPATH=. python scripts/mixvpr/build_vdb_mixvpr.py --config configs/mixvpr/VPAir/build_vdb_mixvpr_VPAir.yaml --batchsize 16

import os
import torch
import argparse

from torch.utils.data import DataLoader

from geoloc.config_parser import load_config, save_config, class_from_config
from geoloc.utils import DEBUG
from geoloc.data.utils import collate_with_geometry


def create_argparse():
    parser = argparse.ArgumentParser(description="MixVPR Vector Database Builder")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--batchsize", type=int, default=16, help="Batch size for processing images"
    )
    return parser


@torch.no_grad()
def build_vdb_mixvpr(config, batchsize=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    savedir = config.savedir
    os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)

    ref_image_dataset = class_from_config(config.dataset)
    ref_image_dataloader = DataLoader(
        ref_image_dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_with_geometry,
    )

    mixvpr_cls, _ = class_from_config(config.mixvpr, instantiate=False)
    mixvpr = mixvpr_cls.load_from_checkpoint(config.mixvpr.checkpoint, strict=True).to(
        device
    )
    mixvpr.eval()

    print("Building database...")
    rotexp_thetas = [0, 90, 180, 270] if config.ROTREF_EXP else [0]
    db = class_from_config(config.vdb)
    db.build(
        ref_image_dataloader, model=mixvpr, rotation_angles=rotexp_thetas, device=device
    )

    print("Saving database...")
    db.save(savedir)
    save_config(config, savedir)


def main():
    parser = create_argparse()
    args = parser.parse_args()
    config = load_config(args.config)
    build_vdb_mixvpr(config, batchsize=args.batchsize)


if __name__ == "__main__":
    main()
