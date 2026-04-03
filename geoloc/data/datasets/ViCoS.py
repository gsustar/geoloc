import os
import json
import torch
import pandas as pd
import geopandas as gpd

from pyproj import Transformer
import torchvision.transforms.functional as TF

from ..utils import get_sorted_imgpaths, load_image
from .GURS import GURSDataset


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


class AformXQueryImages(torch.utils.data.Dataset):
    def __init__(self, query_dir, metadata_path, skip_first_n_frames=400, skip_last_n_frames=400, transforms=None):
        self.query_dir = query_dir
        self.metadata_path = metadata_path
        self.gpkg_path = f"{self.metadata_path[:-4]}_footprints.gpkg"
        self.skip_first_n_frames = skip_first_n_frames
        self.skip_last_n_frames = skip_last_n_frames
        self.crs_transformer = Transformer.from_crs("EPSG:4326", "ESRI:102109", always_xy=True)
        self.crs = "ESRI:102109"

        self.query_images = get_sorted_imgpaths(query_dir)
        self.metadata = pd.read_csv(metadata_path)
        self.geometry_metadata = gpd.read_file(self.gpkg_path)
        if self.skip_first_n_frames > 0:
            self.query_images = self.query_images[self.skip_first_n_frames:]
            self.metadata = self.metadata[self.skip_first_n_frames:].reset_index(drop=True)
            self.geometry_metadata = self.geometry_metadata[self.skip_first_n_frames:].reset_index(drop=True)
        if self.skip_last_n_frames > 0:
            self.query_images = self.query_images[:-self.skip_last_n_frames]
            self.metadata = self.metadata[:-self.skip_last_n_frames].reset_index(drop=True)
            self.geometry_metadata = self.geometry_metadata[:-self.skip_last_n_frames].reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, index):
        img = load_image(self.query_images[index])
        if self.transforms is not None:
            img = self.transforms(img)
        img = img / 255.0
        filename = os.path.basename(self.query_images[index])
        metadata_row = self.metadata.iloc[index]
        geo_metadata_row = self.geometry_metadata.iloc[index]
        # metadata_row = self.metadata[self.metadata["filename"] == filename].iloc[0]
        lon = metadata_row["gps_longitude"]
        lat = metadata_row["gps_latitude"]
        alt = metadata_row["gps_altitude_m"]
        east, north = self.crs_transformer.transform(lon, lat)
        geometry = geo_metadata_row["geometry"]
        return dict(
            image=img,
            filename=filename,
            index=index,
            # lon=lon,
            # lat=lat,
            east=east,
            north=north,
            alt=alt,
            geometry=geometry
        )


class AformXGURSTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, subdir, every_nth_frame=25, transforms=None):
        super().__init__()
        self.border_mappings = {
            "AFormX-flight2-part1": "Savinjska",
            "AFormX-flight3-part1": "Podravska+Koroška",
            "AFormX-flight3-part2": "Savinjska+Podravska",
            "AFormX-flight3-part3": "Pomurska",
        }
        self.every_nth_frame = every_nth_frame
        self.transforms = transforms

        self.gurs_dataset = GURSDataset(
            root="/storage/private/MORS/gurs",
            border=self.border_mappings[subdir],
            tile_size=2000,
            num_same_place=1
        )

        self.afmx_dataset = AformXQueryImages(
            query_dir=f"/storage/private/MORS/AFORMX_GOPRO/frames_3fps_1080p/{subdir}",
            metadata_path=f"/storage/private/MORS/AFORMX_GOPRO/frames_3fps_1080p/{subdir}_3fps_1080p_telemetry.csv",
            skip_first_n_frames=600,
            skip_last_n_frames=600
        )
        self.afmx_dataset = torch.utils.data.Subset(self.afmx_dataset, list(range(0, len(self.afmx_dataset), self.every_nth_frame)))
        # self.utm_transformer = Transformer.from_crs("EPSG:4326", self.gurs_dataset.crs, always_xy=True)

    def __len__(self):
        return len(self.afmx_dataset)

    def __getitem__(self, index):
        query_item = self.afmx_dataset[index]
        query_image = query_item["image"]
        query_image = TF.center_crop(query_image, output_size=1080) # Necessery center_crop, to make image squere
        if self.transforms is not None:
            query_image = self.transforms(query_image)

        gurs_image = self.gurs_dataset.get_data_from_footprint(query_item["geometry"])["image"].squeeze(0)
        if self.transforms is not None:
            gurs_image = self.transforms(gurs_image)

        imgs = torch.stack([query_image, gurs_image], dim=0)
        return dict(
            images=imgs
        )