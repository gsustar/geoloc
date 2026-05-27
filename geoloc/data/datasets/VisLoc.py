import os
import math
import torch
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from rasterio.transform import xy
from rasterio.windows import Window
from shapely.geometry import box

from ..utils import pad_to_size, get_sorted_imgpaths, load_image
import warnings
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")


class VisLocReferenceImages(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        flight_idx,
        tile_size=256,
        stride=None,
        transforms=None,
    ):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.crs = "EPSG:4326"
        self.pxl_res = 0.3
        self.flight_idx = f"{int(flight_idx):02d}"
        self.tif_path = os.path.join(
            self.root, self.flight_idx, f"satellite{self.flight_idx}.tif"
        )
        self.tile_size = tile_size
        self.stride = stride if stride else tile_size

        self.sat_coords = pd.read_csv(
            os.path.join(self.root, "satellite_coordinates_range.csv")
        ).iloc[int(self.flight_idx) - 1]

        # Keep the file open for the lifetime of the dataset
        self.src = rasterio.open(self.tif_path)

        self.tif_width = self.src.width
        self.tif_height = self.src.height
        self.count = self.src.count  # Number of bands
        self.ul = [self.src.bounds.left, self.src.bounds.top]  # Upper left corner
        self.br = [self.src.bounds.right, self.src.bounds.bottom]  # Bottom right corner
        self.transform = self.src.transform

        # Precompute all valid tile windows
        self.tiles = []
        geometries = []
        for x in range(0, self.tif_width, self.stride):
            for y in range(0, self.tif_height, self.stride):
                w = min(self.tile_size, self.tif_width - x)
                h = min(self.tile_size, self.tif_height - y)
                self.tiles.append((x, y, w, h))
                # Convert pixel corners to geographic coords
                # rasterio xy() takes (row, col) — y is row, x is col
                left, top = xy(self.transform, y, x, offset="ul")
                right, bottom = xy(self.transform, y + h, x + w, offset="ul")
                geometries.append(box(left, bottom, right, top))
                # geometries.append(box(x, y, x + w, y + h))
        self.all_windows = gpd.GeoDataFrame(
             geometry=geometries,
             crs=self.crs
        )
        self.num_tiles_width = math.ceil(self.tif_width / self.stride)
        self.num_tiles_height = math.ceil(self.tif_height / self.stride)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        x, y, w, h = self.tiles[index]
        window = Window(x, y, w, h)

        # Use the persistent file handle
        tile = self.src.read(window=window)  # (bands, h, w)
        geometry = box(*self.src.window_bounds(window))

        tile = tile.astype(np.float32)
        tile = torch.from_numpy(tile)  # shape: (C, H, W)

        # Zero-pad if edgetile
        tile = pad_to_size(tile, self.tile_size)

        assert (
            tile.shape[1] == tile.shape[2] == self.tile_size
        ), f"Tile shape mismatch: {tile.shape} != ({self.tile_size}, {self.tile_size})"
        # Apply transforms if provided
        if self.transforms is not None:
            tile = self.transforms(tile)
        tile = tile / 255.0  # Normalize to [0, 1]

        # Get center coordinates (lat/lon or projection units)
        center_col = x + w // 2
        center_row = y + h // 2
        center_x, center_y = xy(self.transform, center_row, center_col)

        return dict(
            image=tile,
            index=index,
            filename="",
            lon=center_x,
            lat=center_y,
            geometry=geometry,
        )

    def get_coords_only(self, index):
        x, y, w, h = self.tiles[index]
        center_col = x + w // 2
        center_row = y + h // 2
        center_x, center_y = xy(self.transform, center_row, center_col)
        return center_x, center_y

    def _get_gt_windows(self, footprint, overlap_threshold=0.5):
        candidates = self.all_windows[self.all_windows.intersects(footprint)]
        intersection_area = candidates.intersection(footprint).area
        overlap_ratio = np.maximum(
            intersection_area / candidates.area,
            intersection_area / footprint.area
        )
        return candidates[overlap_ratio >= overlap_threshold]

    def __del__(self):
        if hasattr(self, "src"):
            self.src.close()


# class VisLocReferenceImages(torch.utils.data.Dataset, ResizeCenterCropMixin):
# 	def __init__(self, root, flight_idx, tile_size=256, stride=None, resize=None, center_crop=None):
# 		super().__init__(resize=resize, center_crop=center_crop)
# 		self.root = root
# 		self.crs = "EPSG:4326"
# 		self.flight_idx = f"{int(flight_idx):02d}"
# 		self.tif_path = os.path.join(self.root, self.flight_idx, f"satellite{self.flight_idx}.tif")
# 		self.tile_size = tile_size
# 		self.stride = stride if stride else tile_size

# 		self.sat_coords = pd.read_csv(
# 			os.path.join(self.root, "satellite_coordinates_range.csv")
# 		).iloc[int(self.flight_idx) - 1]

# 		with rasterio.open(self.tif_path) as src:
# 			self.tif_width = src.width
# 			self.tif_height = src.height
# 			self.count = src.count  # Number of bands
# 			self.ul = [src.bounds.left, src.bounds.top]  # Upper left corner
# 			self.br = [src.bounds.right, src.bounds.bottom]  # Bottom right corner
# 			self.transform = src.transform

# 		# Precompute all valid tile windows
# 		self.tiles = []
# 		for x in range(0, self.tif_width, self.stride):
# 			for y in range(0, self.tif_height, self.stride):
# 				w = min(self.tile_size, self.tif_width - x)
# 				h = min(self.tile_size, self.tif_height - y)
# 				self.tiles.append((x, y, w, h))

# 		self.num_tiles_width = math.ceil(self.tif_width / self.stride)
# 		self.num_tiles_height = math.ceil(self.tif_height / self.stride)

# 	def __len__(self):
# 		return len(self.tiles)

# 	def __getitem__(self, index):
# 		x, y, w, h = self.tiles[index]
# 		window = Window(x, y, w, h)

# 		with rasterio.open(self.tif_path) as src:
# 			tile = src.read(window=window)  # (bands, h, w)
# 			geometry = box(*src.window_bounds(window))

# 		tile = tile.astype(np.float32)
# 		tile = torch.from_numpy(tile)  # shape: (C, H, W)

# 		# Zero-pad if edgetile
# 		tile = pad_to_size(tile, self.tile_size)

# 		assert tile.shape[1] == tile.shape[2] == self.tile_size, (
# 			f"Tile shape mismatch: {tile.shape} != ({self.tile_size}, {self.tile_size})"
# 		)
# 		# Resize if requested
# 		tile = self.resize_centercrop(tile)
# 		tile = tile / 255.0  # Normalize to [0, 1]

# 		# Get center coordinates (lat/lon or projection units)
# 		center_col = x + w // 2
# 		center_row = y + h // 2
# 		center_x, center_y = xy(self.transform, center_row, center_col)

# 		return dict(
# 			image=tile,
# 			index=index,
# 			filename="",
# 			lon=center_x,
# 			lat=center_y,
# 			geometry=geometry,
# 		)


class VisLocQueryImages(torch.utils.data.Dataset):

    def __init__(self, root, flight_idx, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.crs = "EPSG:4326"
        self.flight_idx = f"{int(flight_idx):02d}"
        self.query_images = get_sorted_imgpaths(
            os.path.join(self.root, self.flight_idx, "drone")
        )
        self.query_metadata = pd.read_csv(
            os.path.join(self.root, self.flight_idx, f"{self.flight_idx}.csv")
        )
        self.drone_boxes = gpd.read_file(
            os.path.join(self.root, self.flight_idx, f"drone_boxes.geojson")
        )

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, index):
        img = load_image(self.query_images[index])
        if self.transforms is not None:
            img = self.transforms(img)
        img = img / 255.0
        filename = os.path.basename(self.query_images[index])
        img_metadata = self.query_metadata.iloc[index]
        lon = img_metadata["lon"]  # lon and lat of center of the image
        lat = img_metadata["lat"]
        geometry = self.drone_boxes.iloc[index].geometry
        return dict(
            image=img,
            index=index,
            filename=filename,
            lon=lon,
            lat=lat,
            geometry=geometry,
        )


VISLOC_TRAIN_ROUNDS = ["04", "05", "06", "08", "09", "10", "11"]
VISLOC_TEST_ROUNDS = ["01", "02", "03"]


class VisLocTrainDataset(torch.utils.data.Dataset):

    def __init__(self, root: str, north_align: bool = False, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.crs = "EPSG:4326"
        self.north_align = north_align

        alignment = "north_aligned" if self.north_align else "not_aligned"
        self.all_query_images = []
        self.all_reference_views = []
        self.all_query_metadata = []
        for flight_idx in VISLOC_TRAIN_ROUNDS:
            query_images_path = (
                os.path.join(
                    self.root, flight_idx, "drone_ref_pairs", alignment, "drone"
                )
                if self.north_align
                else os.path.join(self.root, flight_idx, "drone")
            )
            reference_views_path = os.path.join(
                self.root, flight_idx, "drone_ref_pairs", alignment, "reference"
            )
            query_images = get_sorted_imgpaths(query_images_path)
            # query_images = [f"{flight_idx}_" + qi for qi in query_images]
            reference_views = get_sorted_imgpaths(reference_views_path)
            # reference_views = [f"{flight_idx}_" + ri for ri in reference_views]
            self.query_metadata = pd.read_csv(
                os.path.join(self.root, flight_idx, f"{flight_idx}.csv")
            )
            # self.query_metadata["filename"] = f"flight_{flight_idx}_" + self.query_metadata["filename"]
            self.all_query_images = self.all_query_images + query_images
            self.all_reference_views = self.all_reference_views + reference_views
            self.all_query_metadata.append(self.query_metadata)
        self.all_query_metadata = pd.concat(self.all_query_metadata, ignore_index=True)

    def __len__(self):
        return len(self.all_query_images)

    def __getitem__(self, index):
        query = load_image(self.all_query_images[index])
        if self.transforms is not None:
            query = self.transforms(query)
        query = query / 255.0

        ref = load_image(self.all_reference_views[index])
        if self.transforms is not None:
            ref = self.transforms(ref)
        ref = ref / 255.0

        filename = os.path.basename(self.all_query_images[index])
        img_metadata = self.all_query_metadata.iloc[index]
        lon = img_metadata["lon"]  # lon and lat of center of the image
        lat = img_metadata["lat"]

        imgs = torch.stack([query, ref], dim=0)
        return dict(
            images=imgs,
            index=index,
            filename=filename,
            lon=lon,
            lat=lat,
        )
