import os
import torch
import json
import geopandas as gpd
import random

from pyproj import Transformer
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from ..utils import load_image, get_sorted_imgpaths


class GESQueryImages(torch.utils.data.Dataset):
    """
    Query image are images generated from a GES trajectory.
    """

    def __init__(self, root, transforms=None, north_align=False, target_crs="ESRI:102109"):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.crs = "EPSG:4326"
        self.north_align = north_align
        self.target_crs = target_crs

        self.trajectory_name = os.path.basename(root)
        self.footage_dir = os.path.join(root, "footage")

        self.query_images = get_sorted_imgpaths(
            self.footage_dir
        )  # Load query images from the root directory

        self.gpkg_path = os.path.join(
            root, f"{self.trajectory_name}.gpkg"
        )  # additional geometry and camera info
        with open(self.gpkg_path, "r") as f:
            self.geometry = gpd.read_file(self.gpkg_path)
        self.metadata_path = os.path.join(root, f"{self.trajectory_name}.json")
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        if self.target_crs is not None and self.crs != self.target_crs:
            self.swap_crs = True
            self.old_crs = self.crs
            self.geometry.to_crs(self.target_crs, inplace=True)
            self.crs = self.target_crs
            self.crs_transform = Transformer.from_crs(self.old_crs, self.target_crs)

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, index):
        img = load_image(self.query_images[index])
        geometry_data = self.geometry.iloc[index]
        
        filename = os.path.basename(self.query_images[index])
        lon = self.metadata["cameraFrames"][index]["coordinate"]["longitude"]
        lat = self.metadata["cameraFrames"][index]["coordinate"]["latitude"]
        alt = geometry_data["alt"]
        height_above_ground = geometry_data["height_above_ground"]
        geometry = geometry_data["geometry"]
        east, north = None, None
        if self.swap_crs:
            east, north = self.crs_transform.transform(lon, lat)

        roll = geometry_data["roll"]
        pitch = geometry_data["pitch"]
        yaw = geometry_data["yaw"]

        if self.north_align:
            img = TF.rotate(
                img,
                angle=(-yaw * 180.0 / torch.pi).item(),
                interpolation=T.InterpolationMode.BILINEAR,
            )
        if self.transforms is not None:
            img = self.transforms(img)
        img = img / 255.0

        return dict(
            image=img,
            filename=filename,
            index=index,
            lon=lon,
            lat=lat,
            alt=alt,
            east=east,
            north=north,
            height_above_ground=height_above_ground,
            pitch=pitch,
            yaw=yaw,
            geometry=geometry
        )


class GESTrainDataset(torch.utils.data.Dataset):

    def __init__(self, root="/storage/datasets/AerialLoc/Drone2Sat/GES/clean_train", transforms=None, num_gurs_images=1, random_gurs_image=False):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.num_gurs_images = num_gurs_images
        self.random_gurs_image = random_gurs_image
        self.query_images = sorted(os.listdir(root))

    def __len__(self):
        return len(self.query_images)
    
    def __getitem__(self, index):
        imgs = torch.load(os.path.join(self.root, self.query_images[index]))  # [GES image, GURS_image1, GURS_image2, GURS_image3, GURS_image4, GURS_image5] -> GURS_image5 is latest
        ges_img, gurs_imgs = imgs[:1], imgs[1:]

        gurs_img = gurs_imgs[-1:]
        if self.random_gurs_image:
            rix = random.randint(0, len(gurs_imgs)-1)
            gurs_img = gurs_imgs[rix].unsqueeze(0)
        
        imgs = torch.cat([ges_img, gurs_img], dim=0)
        if self.transforms is not None:
            imgs = self.transforms(imgs)
        return dict(
            images=imgs
        )