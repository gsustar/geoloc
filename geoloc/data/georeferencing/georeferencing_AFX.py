import os
import math
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from shapely.geometry import Point

from geoloc.utils import geodetic_to_ecef


# ---------------------------------------------------------------------------
# Camera intrinsics (scaled from 4K base resolution)
# ---------------------------------------------------------------------------

def get_intrinsics(W: int, H: int):
    """
    Return (fx, fy, cx, cy) scaled from the 4K base calibration.

    Base calibration (3840x2160):
        fx0 = fy0 = 1831.2 px
        cx0 = 1920.0,  cy0 = 1080.0
    """
    base_w, base_h = 3840.0, 2160.0
    fx0, fy0, cx0, cy0 = 1831.2, 1831.2, 1920.0, 1080.0
    sx = W / base_w
    sy = H / base_h
    return fx0 * sx, fy0 * sy, cx0 * sx, cy0 * sy


def nadir_ground_radius(W: int, H: int, height_above_ground: float,
                        crop_size: int = None):
    """
    Compute the ground circle radius (metres) for a nadir (straight-down)
    camera at a given height above terrain.

    The radius is the maximum horizontal ground distance from the nadir point
    to any corner ray of the (optionally cropped) image region.

    Parameters
    ----------
    W, H                : full image dimensions in pixels
    height_above_ground : altitude of camera above terrain (metres)
    crop_size           : if given, use only the centre crop_size×crop_size
                          region of the image to compute the radius.
                          Must be ≤ min(W, H).  None = use full image.

    Returns
    -------
    radius_m : float — ground radius in metres
    """
    fx, fy, cx, cy = get_intrinsics(W, H)

    if crop_size is not None:
        # Centre crop: corners of the SxS region centred on (cx, cy)
        half = crop_size / 2.0
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
    else:
        x0, y0 = 0.0,     0.0
        x1, y1 = W - 1.0, H - 1.0

    corners = np.array([
        [x0, y0],   # top-left
        [x1, y0],   # top-right
        [x1, y1],   # bottom-right
        [x0, y1],   # bottom-left
    ], dtype=float)

    # Back-project corner pixels into unit rays (camera frame)
    rays = np.column_stack([
        (corners[:, 0] - cx) / fx,
        (corners[:, 1] - cy) / fy,
        np.ones(4),
    ])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    # Intersect with ground plane at z = height_above_ground and
    # compute horizontal distance from nadir
    radius_m = 0.0
    for ray in rays:
        t = height_above_ground / ray[2]   # ray[2] always > 0 for nadir
        r = math.sqrt((ray[0] * t) ** 2 + (ray[1] * t) ** 2)
        if r > radius_m:
            radius_m = r

    return radius_m


# ---------------------------------------------------------------------------
# DEM sampling
# ---------------------------------------------------------------------------

def sample_dem(dem_path, lon, lat):
    with rasterio.open(dem_path) as src:
        for val in src.sample([(lon, lat)]):
            return float(val[0])


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def create_argparse():
    parser = argparse.ArgumentParser(description="GoPro telemetry georeferencing (circle footprint)")
    parser.add_argument("--csvpath",       type=str,   required=True,
                        help="Path to the telemetry CSV file")
    parser.add_argument("--footage",       type=str,   required=True,
                        help="Path to the footage directory (to read image dimensions)")
    parser.add_argument("--dem",           type=str,   required=True,
                        help="Path to the DEM GeoTIFF")
    parser.add_argument("--width",         type=int,   default=None,
                        help="Image width in pixels (overrides reading from disk)")
    parser.add_argument("--height",        type=int,   default=None,
                        help="Image height in pixels (overrides reading from disk)")
    parser.add_argument("--radius-scale",  type=float, default=1.0,
                        help="Multiplicative scale applied to the computed radius (default: 1.0, e.g. 0.8 = 80%%)")
    parser.add_argument("--radius-offset", type=float, default=0.0,
                        help="Fixed offset in metres subtracted from the computed radius after scaling (default: 0.0)")
    parser.add_argument("--crop-size",     type=int,   default=None,
                        help="Use only the centre SxS pixel region to compute the ground radius "
                             "(e.g. 1080 on a 3840×2160 image). None = use full image (default).")
    parser.add_argument("--rectangle",     action="store_true", default=False,
                        help="Output the bounding rectangle of the circle instead of the circle itself.")
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # PYTHONPATH=. python geoloc/data/georeferencing/georeferencing_gopro.py \
    #   --csvpath /path/to/telemetry.csv \
    #   --footage /path/to/footage \
    #   --dem /storage/private/MORS/SLO_DEM/SLO_DEM.tif
    #   [--crop-size 1080] [--radius-scale 0.9] [--radius-offset 5.0]
    # !!! PYTHONPATH=. python geoloc/data/georeferencing/georeferencing_AFX.py --csvpath /storage/private/MORS/AFORMX_GOPRO/frames_3fps_1080p/AFormX-flight2-part1_3fps_1080p_telemetry.csv --footage /storage/private/MORS/AFORMX_GOPRO/frames_3fps_1080p/AFormX-flight2-part1 --dem /storage/private/MORS/SLO_DEM/SLO_DEM.tif --crop-size 1080 --radius-scale 0.8

    parser = create_argparse()
    args = parser.parse_args()

    df = pd.read_csv(args.csvpath)
    n_frames = len(df)

    # --- Resolve image dimensions once ---
    if args.width and args.height:
        W, H = args.width, args.height
    else:
        import cv2
        first_img_path = os.path.join(args.footage, df.iloc[0]["frame_name"])
        print(f"Reading image dimensions from {first_img_path}...")
        img0 = cv2.imread(first_img_path)
        print(type(img0))
        H, W = img0.shape[:2]

    print(f"Image size: {W}x{H}")

    if args.crop_size is not None:
        if args.crop_size > min(W, H):
            raise ValueError(
                f"--crop-size {args.crop_size} exceeds min(W,H)={min(W, H)}"
            )
        print(f"Using centre crop: {args.crop_size}x{args.crop_size} px")

    frame_info = []
    for i, row in df.iterrows():
        print(f"Frame {i+1}/{n_frames}: {row['frame_name']}")

        lat = row["gps_latitude"]
        lon = row["gps_longitude"]
        alt = row["gps_altitude_m"]

        # --- Height above ground from DEM ---
        ground_elev         = sample_dem(args.dem, lon, lat)
        height_above_ground = alt - ground_elev

        # --- Ground circle radius ---
        radius_m = nadir_ground_radius(W, H, height_above_ground,
                                       crop_size=args.crop_size)
        radius_m = max(0.0, radius_m * args.radius_scale - args.radius_offset)

        # --- Camera position in ECEF (for bookkeeping) ---
        Ex, Ey, Ez = geodetic_to_ecef(lat, lon, alt)

        frame_info.append({
            "frame_index":          row["frame_index"],
            "filename":             row["frame_name"],
            "timestamp_s":          row["timestamp_s"],
            "_lat":                 lat,
            "_lon":                 lon,
            "alt":                  alt,
            "Ex":                   Ex,
            "Ey":                   Ey,
            "Ez":                   Ez,
            "height_above_ground":  height_above_ground,
            "radius_m":             round(radius_m, 3),
            "geometry":             Point(lon, lat),   # replaced with circle below
        })

    # --- Build GeoDataFrame in WGS84, reproject to ESRI:102109 ---
    gdf = gpd.GeoDataFrame(frame_info, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.to_crs("ESRI:102109")

    # --- Projected camera position (east, north) ---
    cam_points_wgs = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [r["_lon"] for r in frame_info],
            [r["_lat"] for r in frame_info],
        ),
        crs="EPSG:4326",
    ).to_crs("ESRI:102109")

    gdf["east"]  = cam_points_wgs.geometry.x.values
    gdf["north"] = cam_points_wgs.geometry.y.values
    # gdf.drop(columns=["_lat", "_lon"], inplace=True)

    # --- Replace point geometry with metric circle (or its bounding rectangle) ---
    gdf["geometry"] = gdf.apply(
        lambda r: Point(r["east"], r["north"]).buffer(r["radius_m"]).envelope
                  if args.rectangle
                  else Point(r["east"], r["north"]).buffer(r["radius_m"]),
        axis=1,
    )

    # --- Save GeoPackage ---
    csv_stem  = os.path.splitext(os.path.basename(args.csvpath))[0]
    out_dir   = os.path.dirname(args.csvpath)
    save_path = os.path.join(out_dir, f"{csv_stem}_footprints.gpkg")

    gdf.to_file(save_path, driver="GPKG")
    print(f"Saved {n_frames} circle footprints to {save_path}")


if __name__ == "__main__":
    main()