import os
import torch
import pandas as pd
from ..utils import load_image, get_sorted_imgpaths


class ALTOReferenceImages(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        round=1,
        split="Train",
        offset="offset_0_None",
        transforms=None,
    ):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.crs = "EPSG:32617"
        self.round = str(round)
        assert self.round in ["1", "2"]
        self.split = split
        assert self.split in ["Train", "Val"]
        self.offset = offset

        self.new_root = os.path.join(self.root, self.round, self.split)
        self.references = get_sorted_imgpaths(
            os.path.join(self.new_root, "reference_images", self.offset)
        )
        self.info = pd.read_csv(os.path.join(self.new_root, "reference.csv"))
        self.info = self.info[self.info["name"].str.contains(self.offset)].reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        img = load_image(self.references[index])
        if self.transforms is not None:
            img = self.transforms(img)
        img = img / 255.0
        filename = self.references[index]
        east = self.info["easting"].iloc[index]
        north = self.info["northing"].iloc[index]
        return dict(
            image=img,
            filename=filename,
            index=index,
            east=east,
            north=north,
        )


class ALTOQueryImages(torch.utils.data.Dataset):

    def __init__(self, root, round=1, split="Val", transforms=None):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.round = str(round)
        assert self.round in ["1", "2"]
        self.split = split
        assert self.split in ["Train", "Val"]
        self.crs = "EPSG:32617"

        self.new_root = os.path.join(self.root, self.round, self.split)
        self.queries = get_sorted_imgpaths(os.path.join(self.new_root, "query_images"))
        self.info = pd.read_csv(os.path.join(self.new_root, "query.csv"))
        self.gt_matches = pd.read_csv(os.path.join(self.new_root, "gt_matches.csv"))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        img = load_image(self.queries[index])
        if self.transforms is not None:
            img = self.transforms(img)
        img = img / 255.0
        filename = self.queries[index]
        east = self.info["easting"].iloc[index]
        north = self.info["northing"].iloc[index]
        alt = self.info["altitude"].iloc[index]
        gt_pos = self.gt_matches["ref_ind"].iloc[index]
        distance_to_gt = self.gt_matches["distance"].iloc[index]
        return dict(
            image=img,
            filename=filename,
            index=index,
            east=east,
            north=north,
            alt=alt,
            gt_pos=gt_pos,
            distance_to_gt=distance_to_gt,
        )

class ALTOTrainDataset(torch.utils.data.Dataset):

    def __init__(self, root, round=1, offset="offset_0_None", transforms=None):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.crs = "EPSG:32617"
        self.round = str(round)
        assert self.round in ["1", "2"]
        self.offset = offset

        self.new_root = os.path.join(self.root, self.round, "Train")
        self.references = get_sorted_imgpaths(
            os.path.join(self.new_root, "reference_images", self.offset)
        )
        self.queries = get_sorted_imgpaths(os.path.join(self.new_root, "query_images"))
        self.gt_matches = pd.read_csv(os.path.join(self.new_root, "gt_matches.csv"))
    
        # self.info = pd.read_csv(os.path.join(self.new_root, "reference.csv"))
        # self.info = self.info[self.info["name"].str.contains(self.offset)].reset_index(
        #     drop=True
        # )
        # self.info_q = pd.read_csv(os.path.join(self.new_root, "query.csv"))

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, index):
        qry_img = load_image(self.queries[index])
        if self.transforms is not None:
            qry_img = self.transforms(qry_img)
        qry_img = qry_img / 255.0

        gt_pos = self.gt_matches["ref_ind"].iloc[index]
        ref_img = load_image(self.references[gt_pos])
        if self.transforms is not None:
            ref_img = self.transforms(ref_img)
        ref_img = ref_img / 255.0

        imgs = torch.stack([qry_img, ref_img], dim=0)
        return dict(
            images=imgs
        )
