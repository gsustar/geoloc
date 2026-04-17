import torch

from geoloc.data.datasets.OrthoLoC import OrthoLocTrainImages
from geoloc.data.datasets.ALTO import ALTOTrainDataset
from geoloc.data.datasets.GES import GESTrainDataset
from geoloc.data.datasets.VisLoc import VisLocTrainDataset
from geoloc.data.datasets.VPAir import VPAirTrainDataset
from geoloc.data.datasets.ViCoS import AformXGURSTrainingDataset

class BigBoyTrainDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None, exclude=None, noaformx=False):
        super().__init__()

        if exclude is None:
            exclude = []
        all_datasets = []

        # ortholoc dataset
        if "ortholoc" not in exclude:
            ortholoc_dataset = OrthoLocTrainImages(
                dataset_dir="/storage/datasets/AerialLoc/OrthoLoC/full",
                mode=2,
                transforms=transforms,
            )
            all_datasets.append(ortholoc_dataset)

        # ALTO dataset
        if "alto" not in exclude:
            alto_dataset = ALTOTrainDataset(
                root="/storage/datasets/AerialLoc/ALTO",
                round=1,
                transforms=transforms
            )
            all_datasets.append(alto_dataset)
        # GES dataset
        if "ges" not in exclude:
            ges_dataset = GESTrainDataset(
                root="/storage/datasets/AerialLoc/Drone2Sat/GES/clean_train",
                num_gurs_images=1,
                random_gurs_image=True,
                transforms=transforms
            )
            all_datasets.append(ges_dataset)
        # VisLoc dataset
        if "visloc" not in exclude:
            visloc_dataset = VisLocTrainDataset(
                root="/storage/datasets/AerialLoc/UAV_VisLoc",
                transforms=transforms,
            )
            all_datasets.append(visloc_dataset)

        # VPAir dataset
        if "vpair" not in exclude:
            vpair_dataset = VPAirTrainDataset(
                root="/storage/datasets/AerialLoc/Drone2Sat/VPAir",
                transforms=transforms,
            )
            all_datasets.append(vpair_dataset)
        # all_datasets = [
        #     ortholoc_dataset,
        #     alto_dataset,
        #     ges_dataset,
        #     visloc_dataset,
        #     vpair_dataset,
        # ]
        aformx_subdirs = [
            "AFormX-flight2-part1",
            "AFormX-flight3-part1",
            # "AFormX-flight3-part2",
            # "AFormX-flight3-part3"
        ]
        if not noaformx or "aformx" not in exclude:
            for subdir in aformx_subdirs:
                aformx_dataset = AformXGURSTrainingDataset(
                    subdir=subdir,
                    every_nth_frame=10,
                    transforms=transforms
                )
                all_datasets.append(aformx_dataset)

        self.dataset = torch.utils.data.ConcatDataset(all_datasets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def bigboy_collate_fn(batch):
    images = [sample["images"] for sample in batch]
    return {"images": torch.stack(images)}
