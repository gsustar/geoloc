import os
import json
import torch

from ..utils import get_sorted_imgpaths, load_image


class ViCoSQueryImages(torch.utils.data.Dataset):
    """
    Query image are image obtained from a ViCoS Drone trajectory.
    """

    def __init__(self, root, transforms=None):
        super().__init__()
        self.root = root
        self.crs = "EPSG:4326"
        self.query_images = get_sorted_imgpaths(root)
        self.transforms = transforms

        # Open metadata file
        metadata_path = os.path.join(root, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, index):
        img = load_image(self.query_images[index])
        if self.transforms is not None:
            img = self.transforms(img)
        img = img / 255.0
        filename = os.path.basename(self.query_images[index])
        img_metadata = self.metadata[self.query_images[index]]
        lon = img_metadata["lon"]
        lat = img_metadata["lat"]
        return dict(
            image=img,
            filename=filename,
            index=index,
            lon=lon,
            lat=lat,
        )
