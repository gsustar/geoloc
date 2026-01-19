import os
import torch
import shapely
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import box
from shapely.prepared import prep
from rasterio.merge import merge
from rasterio.windows import Window

from pyproj import Transformer

class GURSDataset:
    """Base class for GURS datasets."""

    def __init__(
        self,
        root,
        tile_size=1000,
        border="Ljubljana",
        exclude_slo_border_tifs=True,
        num_same_place=3,
        min_year="2011",
    ):
        super().__init__()

        self.root = root
        self.tile_size = tile_size
        self.border = border
        self.exclude_slo_border_tifs = exclude_slo_border_tifs
        self.num_same_place = num_same_place
        self.tif_h, self.tif_w = 6000, 4500
        self.crs = "EPSG:3794"
        self.pxl_res = 0.5

        self._load_border_multipolygon()
        self._load_info_csv(date_filter=min_year)
        if self.exclude_slo_border_tifs:
            self._remove_slo_border_tifs()
        self._load_unique_tifs()
        if self.border != "Slovenija":
            self._remove_outside_border_tifs()
        self._load_discretized_border()
        self.min_east, self.min_north, self.max_east, self.max_north = (
            self.disc_border_polygon.bounds
        )

    def _load_info_csv(self, date_filter: str = None):
        info = pd.read_csv(
            os.path.join(self.root, "gurs_info.csv"), parse_dates=["DAT_POSN"]
        )
        if date_filter is not None:
            date_filter = pd.to_datetime(date_filter)
            info = info[info["DAT_POSN"] >= date_filter]
        self.info = info

    def _load_border_multipolygon(self):
        all_borders_gdf = gpd.read_file(os.path.join(self.root, "borders.geojson"))
        assert (
            self.border in all_borders_gdf["NAZIV"].values
        ), f"Border {self.border} not found in borders.geojson"
        self.border_polygon = all_borders_gdf[
            all_borders_gdf["NAZIV"] == self.border
        ].geometry.values[0]

    def _load_discretized_border(self):
        # recreate the border after removing tifs outside the border
        self.disc_border_polygon = self.unique_tifs.geometry.union_all()

    def _load_unique_tifs(self):
        self.info["TIFNAME"] = self.info["DATOTEKA"].str[:7]
        self.unique_tifs = self.info.drop_duplicates(
            subset="TIFNAME", keep="last"
        ).copy()
        self.unique_tifs["geometry"] = self.unique_tifs.apply(
            lambda row: box(
                row["BOUNDS_LEFT"],
                row["BOUNDS_BOTTOM"],
                row["BOUNDS_RIGHT"],
                row["BOUNDS_TOP"],
            ),
            axis=1,
        )
        self.unique_tifs = gpd.GeoDataFrame(
            self.unique_tifs, geometry="geometry", crs=self.crs
        )

    def _remove_slo_border_tifs(self):
        self.info = self.info[~self.info["IS_ON_BORDER"]]

    def _remove_outside_border_tifs(self):
        self.unique_tifs = self.unique_tifs[
            self.unique_tifs.intersects(self.border_polygon)
        ]

    def _load_window_data(self, src, win):
        return torch.from_numpy(src.read(window=win)).float() / 255.0

    def _get_tifname_matches(self, tifname):
        matches = self.info["DATOTEKA"].str.startswith(tifname)
        return self.info[matches]

    def _get_latest_relevant_tifs_for_window(self, win_bounds):
        return self.unique_tifs[self.unique_tifs.intersects(win_bounds)]

    def _get_all_relevant_tifs_for_window(self, win_bounds):
        latest_relevant_tifs = self._get_latest_relevant_tifs_for_window(win_bounds)
        tifs = self.info[self.info["TIFNAME"].isin(latest_relevant_tifs["TIFNAME"])]
        tifs = dict(tuple(tifs.groupby("TIFNAME")))
        num_rows = min([len(tifs[tifname]) for tifname in tifs.keys()])
        combined_dfs = []
        for i in range(num_rows):
            combined_dfs.append(
                pd.concat(
                    [tifs[tifname].iloc[[i]] for tifname in tifs.keys()],
                    ignore_index=True,
                )
            )
        return combined_dfs


class SequentialGURSDataset(GURSDataset, torch.utils.data.Dataset):
    """Base class for sequential GURS dataset i.e. iterate over all windows inside the border.
    Iteration is done left to right, top to bottom. Note that this is not efficient for training
    """

    def __init__(self, *args, stride=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stride = stride if stride else self.tile_size
        self.num_tiles_height = None
        self.num_tiles_width = None
        self._load_windows()

    def _load_windows(self):
        minx = self.unique_tifs["BOUNDS_LEFT"].min()
        maxx = self.unique_tifs["BOUNDS_RIGHT"].max()
        miny = self.unique_tifs["BOUNDS_BOTTOM"].min()
        maxy = self.unique_tifs["BOUNDS_TOP"].max()

        x_coords = np.arange(minx, maxx, self.stride * self.pxl_res)
        y_coords = np.arange(maxy, miny, -self.stride * self.pxl_res)

        yy, xx = np.meshgrid(y_coords, x_coords)
        x_flat = xx.ravel()
        y_flat = yy.ravel()
        del xx, yy

        self.num_tiles_width = len(x_coords)
        self.num_tiles_height = len(y_coords)

        tile_width = self.tile_size * self.pxl_res
        tile_height = self.tile_size * self.pxl_res

        minxs = x_flat
        maxxs = x_flat + tile_width
        maxys = y_flat
        minys = y_flat - tile_height

        prep_disc_border_polygon = prep(self.disc_border_polygon)
        geometry = shapely.box(minxs, minys, maxxs, maxys)
        self.all_windows = gpd.GeoDataFrame(geometry=geometry, crs=self.crs)
        self.all_windows = self.all_windows[
            prep_disc_border_polygon.contains(self.all_windows.geometry)
        ]
        self.all_windows = self.all_windows.reset_index()

    def __len__(self):
        return len(self.all_windows)


class SequentialWindowGURSDataset(SequentialGURSDataset):
    """Sequentially iterate over all windows inside the border, based on stride and tile size."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        win_bounds = self.all_windows.iloc[index].geometry
        relevant_tifs = self._get_all_relevant_tifs_for_window(win_bounds)
        assert len(relevant_tifs) >= self.num_same_place
        relevant_tifs = relevant_tifs[-self.num_same_place :]
        datas = []
        for tifs in relevant_tifs:
            src_files = [
                rasterio.open(
                    os.path.join(
                        self.root, str(row["YEAR"]), row["AREA_CODE"], row["DATOTEKA"]
                    )
                )
                for _, row in tifs.iterrows()
            ]
            mosaic, _ = merge(src_files, bounds=win_bounds.bounds)
            mosaic = torch.from_numpy(mosaic).float() / 255.0
            assert mosaic.shape == (
                3,
                self.tile_size,
                self.tile_size,
            ), f"Mosaic shape mismatch: {mosaic.shape}"
            for src in src_files:
                src.close()
            datas.append(mosaic)
        datas = torch.stack(datas, dim=0)
        east = (win_bounds.bounds[0] + win_bounds.bounds[2]) / 2
        north = (win_bounds.bounds[1] + win_bounds.bounds[3]) / 2
        return dict(
            image=datas,
            index=index,
            filename="",  # To prevent complications with border images
            east=east,
            north=north,
        )


class GURSReferenceDataset(SequentialGURSDataset):
    """Reference GURS dataset, which returns the latest available image for each window.
    More efficient than SequentialWindowGURSDataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        win_bounds = self.all_windows.iloc[index].geometry
        relevant_tifs = self._get_latest_relevant_tifs_for_window(win_bounds)
        src_files = [
            rasterio.open(
                os.path.join(
                    self.root, str(row["YEAR"]), row["AREA_CODE"], row["DATOTEKA"]
                )
            )
            for _, row in relevant_tifs.iterrows()
        ]
        mosaic, _ = merge(src_files, bounds=win_bounds.bounds)
        mosaic = torch.from_numpy(mosaic).float() / 255.0
        assert mosaic.shape == (
            3,
            self.tile_size,
            self.tile_size,
        ), f"Mosaic shape mismatch: {mosaic.shape}"
        for src in src_files:
            src.close()
        east = (win_bounds.bounds[0] + win_bounds.bounds[2]) / 2
        north = (win_bounds.bounds[1] + win_bounds.bounds[3]) / 2
        return dict(
            image=mosaic,
            filename=relevant_tifs["DATOTEKA"].values[0],
            index=index,
            east=east,
            north=north,
        )

    def get_eastnorth_only(self, index):
        win_bounds = self.all_windows.iloc[index].geometry
        east = (win_bounds.bounds[0] + win_bounds.bounds[2]) / 2
        north = (win_bounds.bounds[1] + win_bounds.bounds[3]) / 2
        return east, north
    
    def get_tile(self, lon, lat, crs="EPSG:4326"):
        east, north = Transformer.from_crs(crs, self.crs, always_xy=True).transform(lon, lat)
        point = shapely.geometry.Point(east, north)
        index = self.all_windows[self.all_windows.geometry.contains(point)].index
        if len(index) == 0:
            return None
        return self.__getitem__(index[0])


class TrainGURSDataset(GURSDataset, torch.utils.data.Dataset):
    """An efficient GURS dataset for training. Uses tif_cache to avoid excessive loading of tif files.
    Similar to SequentialWindowGURSDataset with the exception that (tile_size == stride) and tile_size must be divisible by the tif height and width.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.tif_h % self.tile_size == 0
        ), "TIF height must be divisible by tile size"
        assert (
            self.tif_w % self.tile_size == 0
        ), "TIF width must be divisible by tile size"
        self.tif_windows = self._precompute_windows_for_tif()
        self.num_windows_per_tif = len(self.tif_windows)
        self.num_tifs = len(self.unique_tifs)
        # Define tif cache to avoid reloading tif files
        self.tif_cache = {}
        self.tif_cache_access_count = {}

    def _precompute_windows_for_tif(self):
        self.tif_windows = []
        for y in range(0, self.tif_h - self.tile_size + 1, self.tile_size):
            for x in range(0, self.tif_w - self.tile_size + 1, self.tile_size):
                win = Window(x, y, self.tile_size, self.tile_size)
                self.tif_windows.append(win)
        return self.tif_windows

    def __getitem__(self, index):
        tif_index = index // self.num_windows_per_tif
        win_index = index % self.num_windows_per_tif

        tif = self.unique_tifs.iloc[tif_index]
        win = self.tif_windows[win_index]
        relevant_tifs = self._get_tifname_matches(tif["TIFNAME"])
        assert len(relevant_tifs) >= self.num_same_place
        relevant_tifs = relevant_tifs[-self.num_same_place :]

        datas = []
        for i, r_tif in relevant_tifs.iterrows():
            tiffile = r_tif["DATOTEKA"]
            if tiffile not in self.tif_cache:
                self.tif_cache[tiffile] = rasterio.open(
                    os.path.join(
                        self.root,
                        str(r_tif["YEAR"]),
                        r_tif["AREA_CODE"],
                        r_tif["DATOTEKA"],
                    )
                )
                self.tif_cache_access_count[tiffile] = 0

            data = self.tif_cache[tiffile].read(
                window=win,
            )
            data = torch.from_numpy(data).float() / 255.0
            assert data.shape == (
                3,
                self.tile_size,
                self.tile_size,
            ), f"Data shape mismatch: {data.shape}"
            datas.append(data)

            self.tif_cache_access_count[tiffile] += 1
            if self.tif_cache_access_count[tiffile] >= self.num_windows_per_tif:
                # Close the tif file and remove it from the cache
                self.tif_cache[tiffile].close()
                del self.tif_cache[tiffile]
                del self.tif_cache_access_count[tiffile]

        east = tif["BOUNDS_LEFT"] + (win.col_off + win.width / 2) * self.pxl_res
        north = tif["BOUNDS_TOP"] - (win.row_off + win.height / 2) * self.pxl_res
        return dict(
            image=torch.stack(datas, dim=0),
            filename=tiffile,
            index=index,  # Index is not used in this dataset
            east=east,
            north=north,
        )

    def __len__(self):
        return self.num_tifs * self.num_windows_per_tif

    def teardown(self):
        for tiffile, src in self.tif_cache.items():
            src.close()
        self.tif_cache.clear()

    def __del__(self):
        self.teardown()


class TifAwareShuffleSampler(torch.utils.data.Sampler):
    """A sampler that ensures that windows from the same 'N' TIFs are sampled close together while also ensuring a certain degree of randomness."""

    def __init__(self, num_tifs, num_windows_per_tif, N=3):
        self.num_tifs = num_tifs
        self.num_windows_per_tif = num_windows_per_tif
        self.N = N
        self.shuffled_indices = self._get_shuffled_indices()

    def _get_shuffled_indices(self):
        indices = np.arange(self.num_tifs * self.num_windows_per_tif)
        tif_index_list = np.random.permutation(self.num_tifs)
        shuff_indices = []
        for i in range(0, self.num_tifs, self.N):
            tif_ixs = tif_index_list[i : i + self.N]
            win_ixs = np.array(
                [
                    indices[
                        j
                        * self.num_windows_per_tif : (j + 1)
                        * self.num_windows_per_tif
                    ]
                    for j in tif_ixs
                ]
            ).flatten()
            np.random.shuffle(win_ixs)
            shuff_indices.append(win_ixs)
        return np.concatenate(shuff_indices)

    def __iter__(self):
        self.shuffled_indices = self._get_shuffled_indices()
        return iter(self.shuffled_indices)

    def __len__(self):
        return len(self.shuffled_indices)
