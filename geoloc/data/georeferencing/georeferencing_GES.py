import os
import math
import json
import rasterio
import argparse

import numpy as np
import geopandas as gpd
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt

from natsort import natsorted
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation

from geoloc.utils import geodetic_to_ecef, ecef_to_geodetic


def _get_rotation_matrix(n, theta):
    M = np.array(
        [
            [n[0, 0] ** 2, n[0, 0] * n[1, 0], n[0, 0] * n[2, 0]],
            [n[1, 0] * n[0, 0], n[1, 0] ** 2, n[1, 0] * n[2, 0]],
            [n[2, 0] * n[0, 0], n[2, 0] * n[1, 0], n[2, 0] ** 2],
        ]
    )
    I = np.eye(3)
    M1 = np.array(
        [[0, -n[2, 0], n[1, 0]], [n[2, 0], 0, -n[0, 0]], [-n[1, 0], n[0, 0], 0]]
    )
    R = (1 - math.cos(theta)) * M + math.cos(theta) * I + math.sin(theta) * M1
    return R


def compute_yaw_pitch_roll_from_GES(lat, lon, Rx, Ry, Rz):
    """https://apps.dtic.mil/sti/pdfs/ADA484864.pdf#page=18.47"""
    psi, theta, phi = Rotation.from_euler(
        "XYZ", [Rx, Ry, Rz - 90], degrees=True
    ).as_euler(
        "ZYX", degrees=False
    )  # NOTE: Rz-90 is necessary to obtain NED orientation

    x_0 = np.array([[1, 0, 0]]).T
    y_0 = np.array([[0, 1, 0]]).T
    z_0 = np.array([[0, 0, 1]]).T

    R_psi = _get_rotation_matrix(z_0, psi)
    x_1 = R_psi @ x_0
    y_1 = R_psi @ y_0
    z_1 = R_psi @ z_0

    R_theta = _get_rotation_matrix(y_1, theta)
    x_2 = R_theta @ x_1
    y_2 = R_theta @ y_1
    z_2 = R_theta @ z_1

    R_phi = _get_rotation_matrix(x_2, phi)
    x_3 = R_phi @ x_2
    y_3 = R_phi @ y_2
    z_3 = R_phi @ z_2

    E_0 = np.array([[0, 1, 0]]).T
    N_0 = np.array([[0, 0, 1]]).T
    U_0 = np.array([[1, 0, 0]]).T

    RN0 = _get_rotation_matrix(N_0, math.radians(lon))
    E = RN0 @ E_0
    R_E = _get_rotation_matrix(-E, math.radians(lat))
    N = R_E @ N_0
    U = np.cross(E, N, axis=0)

    x_0 = N
    y_0 = E
    z_0 = -U

    heading = np.arctan2(np.dot(x_3.T, y_0), np.dot(x_3.T, x_0)).squeeze()
    pitch = np.arctan2(
        -np.dot(x_3.T, z_0), np.sqrt(np.dot(x_3.T, x_0) ** 2 + np.dot(x_3.T, y_0) ** 2)
    ).squeeze()
    y_2 = _get_rotation_matrix(z_0, heading) @ y_0
    z_2 = _get_rotation_matrix(y_2, pitch) @ z_0
    roll = np.arctan2(np.dot(y_3.T, z_2), np.dot(y_3.T, y_2)).squeeze()

    heading = np.round(np.rad2deg(heading), 3)
    pitch = np.round(np.rad2deg(pitch), 3)
    roll = np.round(np.rad2deg(roll), 3)
    assert roll == 0, "Roll not supported!!!"

    return heading, pitch, roll


def construct_tangent_plane(lat, lon, ecef_point, height_above_ground):
    n_e = np.array(
        [
            np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
            np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
            np.sin(np.radians(lat)),
        ]
    )
    floor_point = ecef_point - n_e * height_above_ground
    return floor_point, n_e


def compute_plane_ray_intersection(
    plane_point, plane_normal, ray_origin, image_corners_world
):
    """
    Compute the intersection of a ray with a plane.

    Parameters:
            plane_point: A point on the plane (3D vector).
            plane_normal: The normal vector of the plane (3D vector).
            ray_origin: The origin point of the ray (3D vector).
            image_corners_world: The corners of the image in world coordinates (list of 3D vectors).
    Returns:
            intersection_points: List of intersection points (3D vectors)
    """
    intersection_points = []
    for corner in image_corners_world:
        ray_dir = corner - ray_origin
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        denom = np.dot(ray_dir, plane_normal)
        if np.abs(denom) < 1e-6:
            # Ray is parallel to plane
            intersection_points.append(None)
            continue

        t = np.dot(plane_point - ray_origin, plane_normal) / denom
        if t < 0:
            # Intersection behind the camera
            intersection_points.append(None)
            continue

        intersection = ray_origin + t * ray_dir
        intersection = ecef_to_geodetic(*intersection)
        intersection_points.append(intersection)
    return intersection_points


def create_argparse():
    parser = argparse.ArgumentParser(description="GES georeferencing")
    parser.add_argument("--trajpath", type=str, help="Path to the trajectory directory")
    return parser


def main():
    # PYTHONPATH=. python geoloc/data/georeferencing/georeferencing_GES.py --trajpath /storage/datasets/AerialLoc/Drone2Sat/GES/Traj1_Ljubljana_150m_N-align
    parser = create_argparse()
    args = parser.parse_args()
    trajectory_path = args.trajpath
    trajectory_name = os.path.basename(trajectory_path)
    trajectory_json_path = os.path.join(trajectory_path, f"{trajectory_name}.json")

    with open(trajectory_json_path) as f:
        data = json.load(f)

    footage_dir = os.path.join(trajectory_path, "footage")
    filenames = natsorted(os.listdir(footage_dir))
    frame_info = []
    for i, frame in enumerate(data["cameraFrames"]):
        print(f"Frame {i}/{len(data['cameraFrames'])-1}:")
        filename = filenames[i]

        Rx, Ry, Rz = [x for x in frame["rotation"].values()]
        lat, lon, alt = [x for x in frame["coordinate"].values()]
        Ex, Ey, Ez = geodetic_to_ecef(lat, lon, alt)
        ecef_point = np.array([Ex, Ey, Ez])  # camera position in ECEF

        with rasterio.open("/storage/private/MORS/SLO_DEM/SLO_DEM.tif") as src:
            for val in src.sample([(lon, lat)]):
                value = val[0]
        height_above_ground = alt - value

        # Create transformation matrix
        R_mat = pr.matrix_from_euler(np.radians([Rx, Ry, Rz]), 0, 1, 2, extrinsic=False)
        R_transform = pt.transform_from(R=R_mat, p=ecef_point)

        floor_point, n_e = construct_tangent_plane(
            lat, lon, ecef_point, height_above_ground
        )

        # Project image corners to the plane
        verticalFOV_rad = np.radians(80)
        img_height = 2 * height_above_ground * np.tan(verticalFOV_rad / 2)
        aspect_ratio = data["width"] / data["height"]
        img_width = img_height * aspect_ratio
        d = height_above_ground

        # Corners in camera frame (z points forward)
        image_corners_cam = np.array(
            [
                [-img_width / 2, img_height / 2, d],  # top-left
                [img_width / 2, img_height / 2, d],  # top-right
                [img_width / 2, -img_height / 2, d],  # bottom-right
                [-img_width / 2, -img_height / 2, d],  # bottom-left
            ]
        )
        R_cam = R_transform[:3, :3]  # camera rotation
        t_cam = R_transform[:3, 3]  # camera position
        image_corners_world = (R_cam @ image_corners_cam.T).T + t_cam

        intersection_points = compute_plane_ray_intersection(
            plane_point=floor_point,
            plane_normal=n_e,
            ray_origin=t_cam,
            image_corners_world=image_corners_world,
        )
        yaw, pitch, roll = compute_yaw_pitch_roll_from_GES(lat, lon, Rx, Ry, Rz)
        geometry = Polygon(
            [(pt[1], pt[0]) for pt in intersection_points if pt is not None]
        )
        frame_info.append(
            {
                "frame_index": i,
                "filename": filename,
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "Ex": Ex,
                "Ey": Ey,
                "Ez": Ez,
                "Rx": Rx,
                "Ry": Ry,
                "Rz": Rz,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "height_above_ground": height_above_ground,
                "geometry": geometry,
            }
        )

    save_path = os.path.join(trajectory_path, f"{trajectory_name}.gpkg")
    gdf = gpd.GeoDataFrame(frame_info, geometry="geometry", crs="EPSG:4326")
    gdf.to_file(save_path, driver="GPKG")
    print(f"Saved georeferenced trajectory to {save_path}")


if __name__ == "__main__":
    main()
