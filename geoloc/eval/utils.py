import os
import torch
import numpy as np
from matplotlib.table import table
import prettytable as pt


def bytes_to_gb(bytes: int):
    return bytes / (1024**3)


def format_header(header: str, width: int = 40) -> str:
    header_text = f"== {header.upper()} =="
    return header_text.center(width, "=")


def write_resdict_to_file(file, resdict: dict, header: str = None):
    if header:
        print(format_header(header), file=file)
        print("", file=file)
    for key, value in resdict.items():
        print(f"{key:<25}: {value}", file=file)
    print("\n", file=file)


def write_pretty_table(file, field_names, data, align="l", header: str = None):
    table = pt.PrettyTable()
    table.field_names = field_names
    table.align = align
    for row in data:
        formatted_row = []
        for val in row:
            formatted_row.append(f"{val:.2f}" if isinstance(val, float) else val)
        table.add_row(formatted_row)

    if header:
        print(format_header(header), file=file)
        print("", file=file)
    print(table, file=file)
    print("\n", file=file)


# garbage SegVLAD utils
def weighted_borda_count(*ranked_lists_with_scores):
    """
    Merge ranked lists using a weighted Borda Count method where each index's score
    is based on its similarity score rather than its position.
    
    :param ranked_lists_with_scores: Variable number of tuples/lists containing (index, score) pairs.
    :return: A list of indices sorted by their aggregated scores.
    """
    scores = {}
    for ranked_list in ranked_lists_with_scores:
        for index, score in ranked_list:
            if index in scores:
                scores[index] += score
            else:
                scores[index] = score
    sorted_indices = sorted(scores.keys(), key=lambda index: scores[index], reverse=True)
    # sorted_scores = [scores[index] for index in sorted_indices]
    # return sorted_indices, sorted_scores
    return sorted_indices, scores

def segvlad_get_matches(dists, inds, ref_imInds, n=5):
    dists_50 = dists[:, :50]
    inds_50 = inds[:, :50]
    sims_50 = 2 - dists_50

    match_patch = inds_50.T.tolist()
    sims_patch = sims_50.T.tolist()

    # max_seg_preds = func_vpr.get_matches(matches_50,gt,sims_50,segRange2,imInds1,n=5,method="max_seg_topk_wt_borda_Im")
    sims_max = np.max(sims_patch)
    sims_min = np.min(sims_patch)
    sims_patch = (sims_patch - sims_min)/(sims_max-sims_min)
    sims_patch = sims_patch.tolist()

    pair_patch = [list(zip(match_patch[k], sims_patch[k])) for k in range(len(sims_patch))]
    match_patch, scores = weighted_borda_count(*pair_patch)

    seg2im_ind = ref_imInds[match_patch][:, 0]
    imscores = {}
    for i, im_id in enumerate(seg2im_ind):
        seg_score = scores[match_patch[i]]
        if im_id in imscores:
            imscores[im_id] += seg_score
        else:
            imscores[im_id] = seg_score

    # segIdx = np.where(np.bincount(ref_imInds[match_patch]) > 0)[0]
    segIdx = np.where(np.bincount(seg2im_ind) > 0)[0]
    pred = segIdx[np.flip(np.argsort(np.bincount(seg2im_ind)[segIdx])[-n:])]
    pred_score = np.array([imscores[i] for i in pred])
    return pred[None, :], pred_score[None, :]


def inlier_distribution_check(all_num_inliers, threshold = 0.90, max_keypoints=5000):
    pct_inliers_top1 = all_num_inliers[0] / max_keypoints
    diff_top1_vs_10 = (all_num_inliers[0] - all_num_inliers[9]) / max(all_num_inliers)
    combined_metric = pct_inliers_top1 + diff_top1_vs_10
    passed = False
    if combined_metric > threshold:
        passed = True
    return passed, combined_metric


def apply_homography_to_ref_point(
    ref_point,
    homography,
    query_image_shape,
    reference_image_shape,
    meters_per_pixel=0.5,
):
    if homography is None:
        return ref_point

    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3):
        return ref_point

    if query_image_shape is None:
        return ref_point

    qry_h, qry_w = int(query_image_shape[0]), int(query_image_shape[1])
    ref_h, ref_w = int(reference_image_shape[0]), int(reference_image_shape[1])

    query_center = np.asarray([0.5 * (qry_w - 1), 0.5 * (qry_h - 1), 1.0], dtype=np.float64)
    projected = H @ query_center
    if np.abs(projected[2]) < 1e-8:
        return ref_point
    query_center_in_ref = projected[:2] / projected[2]

    ref_center = np.asarray([0.5 * (ref_w - 1), 0.5 * (ref_h - 1)], dtype=np.float64)
    delta_px = query_center_in_ref - ref_center

    delta_m_x = delta_px[0] * float(meters_per_pixel)
    delta_m_y = -delta_px[1] * float(meters_per_pixel)

    ref_point = np.asarray(ref_point, dtype=np.float64).reshape(2)
    predicted = ref_point + np.asarray([delta_m_x, delta_m_y], dtype=np.float64)
    return predicted