import math
import numpy as np

from pyproj import CRS, Geod
from shapely.geometry.polygon import Polygon
from geoloc.utils import crs_transform


def distance_between_points(qry_point: tuple, ref_point: tuple, crs: str):
    crs = CRS.from_user_input(crs)
    qx, qy = qry_point
    rx, ry = ref_point

    if crs.is_projected:
        return math.sqrt((qx - rx) ** 2 + (qy - ry) ** 2)
    elif crs.is_geographic:
        geod = Geod(ellps="WGS84")
        _, _, dist = geod.inv(qx, qy, rx, ry)
        return dist
    else:
        raise ValueError("CRS must be either projected or geographic.")


def polygon_intersection_metrics(
    qry_polygon: Polygon, ref_polygon: Polygon, tp_threshold=0.25
):
    inter = qry_polygon.intersection(ref_polygon)
    qry_ioa = inter.area / qry_polygon.area
    ref_ioa = inter.area / ref_polygon.area
    area_imbalance = max(qry_polygon.area, ref_polygon.area) / min(
        qry_polygon.area, ref_polygon.area
    )
    is_tp = max(qry_ioa, ref_ioa) >= tp_threshold
    return qry_ioa, ref_ioa, area_imbalance, is_tp


def calculate_distances(
    qry, benchmark_top_k, inds, qry_image_dataset, ref_image_dataset
):
    qry_point = qry["lon"], qry["lat"]
    qry_point = crs_transform(qry_point, qry_image_dataset.crs, ref_image_dataset.crs)
    ref_points = []
    for cpr_i in range(max(benchmark_top_k)):
        if cpr_i >= inds.shape[1]:
            break
        ref = ref_image_dataset[(inds[0, cpr_i] % len(ref_image_dataset)).item()]
        ref_points.append((ref["lon"], ref["lat"]))
    gdists = np.array(
        [
            distance_between_points(qry_point, ref_point, ref_image_dataset.crs)
            for ref_point in ref_points
        ]
    )
    return gdists, ref_points


def calculate_intersections(qry, benchmark_top_k, inds, ref_image_dataset):
    gt_pos = []
    intersection_tps = []
    for cpr_i in range(max(benchmark_top_k)):
        if cpr_i >= inds.shape[1]:
            break
        ref_ix = (inds[0, cpr_i] % len(ref_image_dataset)).item()
        ref = ref_image_dataset[ref_ix]
        qry_ioa, ref_ioa, area_imbalance, is_tp = polygon_intersection_metrics(
            qry["geometry"],
            ref["geometry"],
        )
        if is_tp:
            gt_pos.append(ref_ix)
        intersection_tps.append(is_tp)
    return gt_pos, intersection_tps


# def calculate_intersections(qry, benchmark_top_k, inds, ref_image_dataset):
# 	gt_pos = []
# 	intersection_tps = []
# 	qry_geom = qry["geometry"]
# 	qry_bounds = qry_geom.bounds

# 	for cpr_i in range(max(benchmark_top_k)):
# 		if cpr_i >= inds.shape[1]:
# 			break
# 		ref_ix = (inds[0, cpr_i] % len(ref_image_dataset)).item()
# 		ref = ref_image_dataset[ref_ix]
# 		ref_geom = ref["geometry"]

# 		ref_bounds = ref_geom.bounds
# 		if (qry_bounds[2] < ref_bounds[0] or qry_bounds[0] > ref_bounds[2] or
# 			qry_bounds[3] < ref_bounds[1] or qry_bounds[1] > ref_bounds[3]):
# 			intersection_tps.append(False)
# 			continue

# 		qry_ioa, ref_ioa, area_imbalance, is_tp = polygon_intersection_metrics(
# 			qry_geom, ref_geom,
# 		)
# 		if is_tp:
# 			gt_pos.append(ref_ix)
# 		intersection_tps.append(is_tp)

# 	return gt_pos, intersection_tps
