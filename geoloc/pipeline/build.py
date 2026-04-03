# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. DEBUG=0 python scripts/build.py --config configs/model/dataset/build_config.yaml --batchsize 16

import os
import torch
import argparse

from torch.utils.data import DataLoader

from geoloc.config_parser import load_config, save_config, class_from_config
from geoloc.utils import DEBUG, load_model
from geoloc.data.utils import collate_with_geometry


def create_argparse():
    parser = argparse.ArgumentParser(
        description="Vector Database Builder"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the build configuration file"
    )
    parser.add_argument(
        "--batchsize", type=int, default=1, help="Batch size for processing images"
    )
    return parser

@torch.no_grad()
def build_vdb(config, batchsize: int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vdbdir = config.savedir
    os.makedirs(vdbdir, exist_ok=True if DEBUG > 0 else False)
    
    ref_image_dataset = class_from_config(config.dataset)
    ref_image_dataloader = DataLoader(
        ref_image_dataset,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_with_geometry,
    )
    model = load_model(config).eval().to(device)
    model.compile()
    
    print("Building database...")
    vdb = class_from_config(config.vdb)
    vdb.build(
        ref_image_dataloader,
        model=model,
        rotation_angles=[0, 90, 180, 270] if getattr(config, "ROTREF_EXP", False) else None,
        device=device,
        save_salad_matrix=getattr(config, "SAVE_REF_SALAD_MATRIX", False),
        salad_matrix_savedir=os.path.join(vdbdir, "ref_salad_matrices")
    )

    print("Saving database...")
    vdb.save(vdbdir)
    save_config(config, vdbdir, prefix="build")
    
    print(f"Database saved to {vdbdir}")
    print(f"  - Database size: {vdb.size()}")
    print(f"  - Vector dimension: {vdb.vdim}")


def main():
    parser = create_argparse()
    args = parser.parse_args()
    config = load_config(args.config)
    build_vdb(config, batchsize=args.batchsize)


if __name__ == "__main__":
    main()
