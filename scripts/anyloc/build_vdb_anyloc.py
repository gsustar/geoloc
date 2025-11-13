# CUDA_VISIBLE_DEVICES=1 DEBUG=0 PYTHONPATH=. python scripts/anyloc/build_vdb_anyloc.py --config configs/anyloc/VPAir/build_vdb_anyloc_VPAir.yaml --batchsize 16

import os
import torch
import argparse

from torch.utils.data import DataLoader

from geoloc.config_parser import load_config, save_config, class_from_config
from geoloc.utils import DEBUG
from geoloc.data.utils import collate_with_geometry


def create_argparse():
    parser = argparse.ArgumentParser(description="AnyLoc Vector Database Builder")
    parser.add_argument(
        "--vdbdir", type=str, help="Path to the vector database directory"
    )
    parser.add_argument(
        "--build_config", type=str, help="Path to the build configuration file"
    )
    parser.add_argument(
        "--batchsize", type=int, default=16, help="Batch size for processing images"
    )
    return parser


@torch.no_grad()
def build_vdb_anyloc(vdbdir, build_config, batchsize=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # savedir = vdbdir
    # os.makedirs(savedir, exist_ok=True if DEBUG > 0 else False)
    fit_config = load_config(os.path.join(vdbdir, "fit_config.yaml"))

    ref_image_dataset = class_from_config(build_config.dataset)
    ref_image_dataloader = DataLoader(
        ref_image_dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_with_geometry,
    )

    anyloc = class_from_config(fit_config.anyloc).to(device)
    anyloc.load_vlad(vdbdir)
    anyloc.load_pca(vdbdir)

    print("Building database...")
    rotexp_thetas = [0, 90, 180, 270] if build_config.ROTREF_EXP else [0]
    db = class_from_config(build_config.vdb)
    db.build(
        ref_image_dataloader, model=anyloc, rotation_angles=rotexp_thetas, device=device
    )

    print("Saving database...")
    db.save(vdbdir)
    save_config(build_config, vdbdir, prefix="build")


def main():
    parser = create_argparse()
    args = parser.parse_args()
    fit_config = load_config(args.fit_config)
    build_config = load_config(args.build_config)
    build_vdb_anyloc(fit_config, build_config, batchsize=args.batchsize)


if __name__ == "__main__":
    main()
