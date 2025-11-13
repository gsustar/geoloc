import os
import torch
import json
import torchvision.transforms.functional as F

from ..utils import load_image, get_sorted_imgpaths
from .base import ResizeCenterCropMixin


class GESQueryImages(torch.utils.data.Dataset, ResizeCenterCropMixin):
    """
    Query image are images generated from a GES trajectory.
    """

    def __init__(self, root, resize=None, center_crop=None):
        super().__init__(resize=resize, center_crop=center_crop)
        self.root = root
        self.crs = "EPSG:4326"

        self.trajectory_name = os.path.basename(os.path.dirname(root))
        self.footage_dir = os.path.join(root, "footage")

        self.query_images = get_sorted_imgpaths(
            self.footage_dir
        )  # Load query images from the root directory

        self.metadata_path = os.path.join(root, f"{self.trajectory_name}.json")
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, index):
        img = load_image(self.query_images[index])
        img = self.resize_centercrop(img)
        img = img / 255.0
        filename = os.path.basename(self.query_images[index])
        lon = self.metadata["cameraFrames"][index]["coordinate"]["longitude"]
        lat = self.metadata["cameraFrames"][index]["coordinate"]["latitude"]
        return dict(
            image=img,
            filename=filename,
            index=index,
            lon=lon,
            lat=lat,
        )
