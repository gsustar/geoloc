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
    diff_top1_vs_10 = (all_num_inliers[0] - all_num_inliers[9]) / (max(all_num_inliers) + 1e-8)
    combined_metric = pct_inliers_top1 + diff_top1_vs_10
    passed = False
    if combined_metric > threshold:
        passed = True
    return passed, combined_metric


def get_ref_points(
    inds, ref_image_dataset, benchmark_top_k
):
    ref_points = []
    for cpr_i in range(max(benchmark_top_k)):
        if cpr_i >= inds.shape[1]:
            break
        
        coords = ref_image_dataset.get_coords_only((inds[0, cpr_i] % len(ref_image_dataset)).item())
        ref_points.append(coords)
    return ref_points