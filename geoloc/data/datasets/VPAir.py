import os
import yaml
import torch
import numpy as np
import pandas as pd

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from ..utils import load_image, get_sorted_imgpaths


def _load_poses(root):
    return pd.read_csv(os.path.join(root, "poses.csv"))


def _load_camera_calibration(root):
    with open(os.path.join(root, "camera_calibration.yaml"), "r") as f:
        camera_calibration = yaml.safe_load(f)
    return camera_calibration


def _load_distractors(root):
    return get_sorted_imgpaths(os.path.join(root, "distractors"))


def _load_reference_views(root, canonical_ori=False):
    refdir = "reference_views_can" if canonical_ori else "reference_views"
    return get_sorted_imgpaths(os.path.join(root, refdir))


def _load_queries(root, north_aligned=False, canonical_ori=False):
    assert not (north_aligned and canonical_ori)
    querydir = "queries"
    if canonical_ori:
        querydir = "queries_can"
    elif north_aligned:
        querydir = "queries_north_aligned"
    return get_sorted_imgpaths(os.path.join(root, querydir))


def _load_gt_pos(root):
    return np.load(os.path.join(root, "vpair_gt.npy"), allow_pickle=True)


class VPAirReferenceImages(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        include_distractors: bool = True,
        percent_distractors: float = 1.0,
        canonical_ori: bool = False,
        transforms=None,
    ):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.crs = "EPSG:4326"
        self.include_distractors = include_distractors
        self.percent_distractors = percent_distractors

        self.distractors = _load_distractors(root)
        self.reference_views = _load_reference_views(root, canonical_ori=canonical_ori)
        self.num_distractors = len(self.distractors)
        self.num_reference_views = len(self.reference_views)
        self.poses = _load_poses(root)

        # Optionally include distractors in reference images
        self.reference_images = self.reference_views.copy()
        if self.include_distractors:
            self.num_distractors = int(self.percent_distractors * self.num_distractors)
            self.distractors = self.distractors[:self.num_distractors]
            self.reference_images.extend(self.distractors)
        

        del self.distractors
        del self.reference_views

    def __len__(self):
        return len(self.reference_images)

    def __getitem__(self, index):
        is_distractor = index >= self.num_reference_views
        img = load_image(self.reference_images[index])
        if self.transforms is not None:
            img = self.transforms(img)
        img = img / 255.0
        filename = os.path.basename(self.reference_images[index])
        # Distractors do not have pose information
        lon = 0.0 if is_distractor else self.poses["lon"].iloc[index]
        lat = 0.0 if is_distractor else self.poses["lat"].iloc[index]
        alt = 0.0 if is_distractor else self.poses["altitude"].iloc[index]
        return dict(
            image=img,
            filename=filename,
            index=index,
            lon=lon,
            lat=lat,
            alt=alt,
            is_distractor=is_distractor,
            gt_pos=-1,
        )


class VPAirQueryImages(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        soft_positive_offset: int = 3,
        north_align=False,
        canonical_ori=False,
        transforms=None,
    ):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.crs = "EPSG:4326"
        self.north_align = north_align
        self.canonical_ori = canonical_ori
        assert not (north_align and canonical_ori)
        # self.queries = _load_queries(
        #     root, north_aligned=north_align, canonical_ori=canonical_ori
        # )
        self.queries = _load_queries(
            root, canonical_ori=canonical_ori
        )
        self.poses = _load_poses(root)
        # self.gt_pos = _load_gt_pos(root)
        self.soft_positive_offset = soft_positive_offset

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        img = load_image(self.queries[index])
        lon = self.poses["lon"].iloc[index]
        lat = self.poses["lat"].iloc[index]
        alt = self.poses["altitude"].iloc[index]

        roll = self.poses["roll"].iloc[index]
        pitch = self.poses["pitch"].iloc[index]
        yaw = self.poses["yaw"].iloc[index] # This is the CW rotation of drone image wrt north, in radians

        if self.north_align:
            img = TF.rotate(
                img,
                angle=(-yaw * 180.0 / torch.pi).item(),
                interpolation=T.InterpolationMode.BILINEAR,
            )
        if self.transforms is not None:
            img = self.transforms(img)
        img = img / 255.0
        filename = os.path.basename(self.queries[index])

        gt_pos = np.unique(
            np.clip(
                list(
                    range(
                        # (index + 1) - self.soft_positive_offset,
                        # (index + 1) + self.soft_positive_offset + 1
                        index - self.soft_positive_offset,
                        index + self.soft_positive_offset + 1,
                    )
                ),
                0,
                len(self.queries),
            )
        )
        return dict(
            image=img,
            filename=filename,
            index=index,
            lon=lon,
            lat=lat,
            alt=alt,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            gt_pos=gt_pos,
        )


VPAIR_TRAIN_TEST_SPLIT = 0.7
class VPAirTrainDataset(torch.utils.data.Dataset):

    def __init__(
        self, root: str, north_align: bool = False,
        transforms=None, split: str = "train",
    ):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.north_align = north_align
        # self.queries = _load_queries(root, north_aligned=north_align)
        self.queries = _load_queries(root)
        self.reference_views = _load_reference_views(root)

        train_cut = int(np.floor(len(self.queries) * VPAIR_TRAIN_TEST_SPLIT))
        if split == "train":
            self.queries = self.queries[:train_cut]
            self.reference_views = self.reference_views[:train_cut]
            self.poses = _load_poses(root).iloc[:train_cut].reset_index(drop=True)
        elif split == "test":
            self.queries = self.queries[train_cut:]
            self.reference_views = self.reference_views[train_cut:]
            self.poses = _load_poses(root).iloc[train_cut:].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")


    def __len__(self):
        assert len(self.queries) == len(self.reference_views)
        return len(self.queries)

    def __getitem__(self, index):
        query = load_image(self.queries[index])
        lon = self.poses["lon"].iloc[index]
        lat = self.poses["lat"].iloc[index]
        alt = self.poses["altitude"].iloc[index]

        roll = self.poses["roll"].iloc[index]
        pitch = self.poses["pitch"].iloc[index]
        yaw = self.poses["yaw"].iloc[index]

        if self.north_align:
            # yaw = self.poses["yaw"].iloc[index]
            query = TF.rotate(
                query,
                angle=(-yaw * 180.0 / torch.pi).item(),
                interpolation=T.InterpolationMode.BILINEAR,
            )
        
        if self.transforms is not None:
            query = self.transforms(query)
        query = query / 255.0

        ref = load_image(self.reference_views[index])
        if self.transforms is not None:
            ref = self.transforms(ref)
        ref = ref / 255.0

        imgs = torch.stack([query, ref], dim=0)
        return dict(
            images=imgs,  # [N, C, H, W], where N is the number of same-place images (2 here)
            index=index,
            lon=lon,
            lat=lat,
            alt=alt,
            roll=roll,
            pitch=pitch,
            yaw=yaw, # Make sure yaw is in radians!!!
        )

class VPAirTrainDatasetQueryImages(VPAirQueryImages):

    def __init__(
        self,
        root: str,
        soft_positive_offset: int = 3,
        north_align=False,
        canonical_ori=False,
        transforms=None,
    ):
        super().__init__(
            root=root,
            soft_positive_offset=soft_positive_offset,
            north_align=north_align,
            canonical_ori=canonical_ori,
            transforms=transforms,
        )
        train_cut = int(np.floor(len(self.queries) * VPAIR_TRAIN_TEST_SPLIT))
        self.queries = self.queries[:train_cut]
        self.poses = self.poses.iloc[:train_cut].reset_index(drop=True)


class VPAirTrainDatasetReferenceImages(VPAirReferenceImages):

    def __init__(
        self,
        root: str,
        include_distractors: bool = True,
        canonical_ori: bool = False,
        transforms=None,
    ):
        super().__init__(
            root=root,
            include_distractors=include_distractors,
            canonical_ori=canonical_ori,
            transforms=transforms,
        )
        train_cut = int(np.floor(self.num_reference_views * VPAIR_TRAIN_TEST_SPLIT))
        self.num_reference_views = train_cut
        self.reference_images = self.reference_images[:train_cut]
        self.poses = self.poses.iloc[:train_cut].reset_index(drop=True)

class VPAirTestDatasetQueryImages(VPAirQueryImages):

    def __init__(
        self,
        root: str,
        soft_positive_offset: int = 3,
        north_align=False,
        canonical_ori=False,
        transforms=None,
    ):
        super().__init__(
            root=root,
            soft_positive_offset=soft_positive_offset,
            north_align=north_align,
            canonical_ori=canonical_ori,
            transforms=transforms,
        )
        train_cut = int(np.floor(len(self.queries) * VPAIR_TRAIN_TEST_SPLIT))
        self.queries = self.queries[train_cut:]
        self.poses = self.poses.iloc[train_cut:].reset_index(drop=True)


class VPAirTestDatasetReferenceImages(VPAirReferenceImages):

    def __init__(
        self,
        root: str,
        include_distractors: bool = True,
        percent_distractors: float = 1.0,
        canonical_ori: bool = False,
        transforms=None,
    ):
        super().__init__(
            root=root,
            include_distractors=include_distractors,
            percent_distractors=percent_distractors,
            canonical_ori=canonical_ori,
            transforms=transforms,
        )
        train_cut = int(np.floor(self.num_reference_views * VPAIR_TRAIN_TEST_SPLIT))
        self.num_reference_views = self.num_reference_views - train_cut
        self.reference_images = self.reference_images[train_cut:]
        self.poses = self.poses.iloc[train_cut:].reset_index(drop=True)
