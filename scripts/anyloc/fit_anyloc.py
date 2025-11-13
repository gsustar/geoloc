# CUDA_VISIBLE_DEVICES=1 DEBUG=0 PYTHONPATH=. python scripts/anyloc/build_vdb_anyloc.py --config configs/anyloc/VPAir/build_vdb_anyloc_VPAir.yaml --batchsize 16

import os
import pickle
import torch
import einops
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader

from geoloc.config_parser import load_config, save_config, class_from_config
from geoloc.utils import DEBUG
from geoloc.data.utils import collate_with_geometry


def create_argparse():
    parser = argparse.ArgumentParser(description="AnyLoc Vector Database Builder")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--batchsize", type=int, default=16, help="Batch size for processing images"
    )
    return parser


@torch.no_grad()
def fit_anyloc(config, batchsize=16):
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

    anyloc = class_from_config(config.anyloc).to(device)

    print("Extracting features with backbone...")
    features = []
    for i, batch in enumerate(tqdm(ref_image_dataloader)):
        if DEBUG > 1 and i > 20:
            break
        if i % config.fitstep != 0:
            continue
        image = batch["image"].to(device)
        x = anyloc.backbone(image)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        features.append(x.detach())
    features = torch.vstack(features)

    if config.vlad_prefitted:
        anyloc.vlad.load_c_centers(config.vlad_prefitted)
        features = anyloc.vlad.generate_multi(features)
        anyloc.vlad.save()  # Manually save `c_centers` (usually this happens during `fit`)
    else:
        print("Fitting VLAD features...")
        assert anyloc.vlad.cache_dir is not None
        features = anyloc.vlad.fit_and_generate(features)

    if anyloc.pca is not None:
        print("Fitting PCA...")
        features = anyloc.pca.fit_transform(features.cpu())
        with open(os.path.join(savedir, "pca.pkl"), "wb") as f:
            pickle.dump(anyloc.pca, f)
        print(f"PCA components: {anyloc.pca.components_.shape}")

    save_config(config, savedir, prefix="fit")


def main():
    parser = create_argparse()
    args = parser.parse_args()
    config = load_config(args.config)
    fit_anyloc(config, batchsize=args.batchsize)


if __name__ == "__main__":
    main()
