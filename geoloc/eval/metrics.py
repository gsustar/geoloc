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


def first_true_index(bool_arr):
    """Return the 0-based index of the first True, or None if none."""
    if len(bool_arr) == 0:
        return None
    idx = np.argmax(bool_arr)
    return int(idx) if bool_arr[idx] else None

def safe_rank1(tp_flags):
    """Return 1-based rank of first TP, or None if none."""
    idx0 = first_true_index(tp_flags)
    return (idx0 + 1) if idx0 is not None else None

def rank1_or_sentinel(tp_flags, sentinel=999):
    """Return rank1 (1-based) or a large sentinel for failures (sortable)."""
    r1 = safe_rank1(tp_flags)
    return sentinel if r1 is None else r1

def reciprocal_rank_from_rank1(rank1):
    """Reciprocal Rank from 1-based rank; 0.0 if None."""
    return 0.0 if (rank1 is None) else 1.0 / rank1

def tp_at_k(tp_flags, k):
    """Count of TPs among the top-k."""
    k = min(k, len(tp_flags))
    return int(np.sum(tp_flags[:k]))

def hit_at_k(tp_flags, k):
    """Binary hit among top-k (your Recall@K)."""
    k = min(k, len(tp_flags))
    return 1 if np.any(tp_flags[:k]) else 0

def average_precision(tp_flags):
    """
    AP for a single ranked list: mean precision at TP ranks.
    If no TPs, returns 0.0.
    """
    tp_positions = np.where(tp_flags)[0]
    if tp_positions.size == 0:
        return 0.0

    precisions = []
    tp_cum = 0
    for i, is_tp in enumerate(tp_flags):
        if is_tp:
            tp_cum += 1
            precisions.append(tp_cum / (i + 1))
    return float(np.mean(precisions))



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
