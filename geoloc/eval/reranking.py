import torch
import numpy as np

from geoloc.data.utils import collate_with_geometry


def estimate_homography(kptsA, kptsB, homography_estimator):
    Hs = []
    masks = []
    for kA, kB in zip(kptsA, kptsB):
        valid_mask = (kA[:, 0] >= 0) & (kA[:, 1] >= 0) & (kB[:, 0] >= 0) & (kB[:, 1] >= 0)
        kA = kA[valid_mask]
        kB = kB[valid_mask]
        H, mask = homography_estimator(kA.float(), kB.float())
        Hs.append(H.squeeze(0))
        mask = torch.nn.functional.pad(mask, (0, 0, 0, kptsA.shape[1] - mask.shape[0]), value=False)
        masks.append(mask.squeeze(0))
    num_inliers = torch.stack([m.sum() for m in masks])
    if len(Hs) > 0:
        Hs = torch.stack(Hs, dim=0)
    if len(masks) > 0:
        masks = torch.stack(masks, dim=0)
    return Hs, masks, num_inliers

def estimate_pose_pnp(kptsA, kptsB, K, dems, geoms, pnp_solver):
    R = []
    tvec = []
    masks = []
    covs = []
    for kA, kB, dem, geom in zip(kptsA, kptsB, dems, geoms):
        valid_mask = (kA[:, 0] >= 0) & (kA[:, 1] >= 0) & (kB[:, 0] >= 0) & (kB[:, 1] >= 0)
        kA_f = kA[valid_mask]
        kB_f = kB[valid_mask]
        R_i, tvec_i, mask_i, cov_i = pnp_solver(kA_f.float(), kB_f.float(), K, dem, geom)
        R.append(R_i)
        tvec.append(tvec_i)
        covs.append(cov_i)
        # scatter the inlier mask back to the original keypoint rows
        full_mask = torch.zeros((kA.shape[0], 1), dtype=torch.bool)
        full_mask[valid_mask] = mask_i.to(full_mask.device)
        masks.append(full_mask)
    num_inliers = torch.stack([m.sum() for m in masks])
    if len(R) > 0:
        R = torch.stack(R, dim=0)
    if len(tvec) > 0:
        tvec = torch.stack(tvec, dim=0)
    if len(masks) > 0:
        masks = torch.stack(masks, dim=0)
    return R, tvec, masks, num_inliers, covs


def rerank(
    matcher, ransac, qry_image, ref_image_dataset, inds, ransac_mode="homography", K=None, dists=None, device=None, batch_size=1,
    bootstrap_ransac=None
):
    assert matcher is not None, "Matcher model must be provided"
    # if isinstance(inds, np.ndarray):
    #     inds = torch.from_numpy(inds)
    if dists is None:
        dists = torch.ones_like(inds).float() * -1.0

    if device is None:
        device = matcher.device

    if qry_image.ndim == 3:
        qry_image = qry_image.unsqueeze(0)

    mapped_inds = [int(idx % len(ref_image_dataset)) for idx in inds[0]]
    subset = torch.utils.data.Subset(ref_image_dataset, mapped_inds)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_with_geometry)

    all_num_inliers = []
    all_homographies = []
    all_Rs = []
    all_tvecs = []
    # all_covs = []
    all_num_outliers = []
    all_kptsA = []
    all_kptsB = []
    all_masks = []
    # # bootstrap only ever needs the single highest-inlier candidate's DEM/geometry,
    # # so track that running best instead of holding every candidate's DEM tile
    # # in memory
    # best_dem = None
    # best_geom = None
    # best_num_inliers = -1
    for batch in dataloader:
        ref_batch = batch["image"].to(device)
        qry_batch = qry_image.expand(ref_batch.shape[0], -1, -1, -1)

        kptsA, kptsB = matcher(qry_batch, ref_batch, repeated_qry_optimization=True)
        if ransac_mode == "homography":
            Hs, masks, num_inliers = estimate_homography(kptsA, kptsB, ransac)
            all_homographies.append(Hs)
        elif ransac_mode == "pnp":
            Rs, tvecs, masks, num_inliers, covs = estimate_pose_pnp(kptsA, kptsB, K, batch["dem"], batch["geometry"], ransac)
            all_Rs.append(Rs)
            all_tvecs.append(tvecs)
            # all_covs.extend(covs)
            # if bootstrap_ransac is not None:
            #     batch_best_idx = int(torch.argmax(num_inliers).item())
            #     batch_best_ninl = int(num_inliers[batch_best_idx].item())
            #     if batch_best_ninl > best_num_inliers:
            #         best_num_inliers = batch_best_ninl
            #         # .clone(): batch["dem"] is a stacked tensor, so an unclonded
            #         # slice would keep the whole batch's underlying storage alive
            #         best_dem = batch["dem"][batch_best_idx].clone()
            #         best_geom = batch["geometry"][batch_best_idx]
        else:
            raise ValueError(f"Invalid ransac_mode: {ransac_mode}")
        
        num_outliers = kptsA.shape[1] - num_inliers
        all_num_inliers.append(num_inliers)
        all_num_outliers.append(num_outliers)
        all_kptsA.append(kptsA)
        all_kptsB.append(kptsB)
        all_masks.append(masks)

    all_num_inliers = torch.cat(all_num_inliers, dim=0).cpu()
    all_num_outliers = torch.cat(all_num_outliers, dim=0).cpu()
    all_kptsA = torch.cat(all_kptsA, dim=0).cpu()
    all_kptsB = torch.cat(all_kptsB, dim=0).cpu()
    all_masks = torch.cat(all_masks, dim=0).cpu()
    if ransac_mode == "homography":
        all_homographies = torch.cat(all_homographies, dim=0).cpu()
    elif ransac_mode == "pnp":
        all_Rs = torch.cat(all_Rs, dim=0).cpu()
        all_tvecs = torch.cat(all_tvecs, dim=0).cpu()

    sorted_order = torch.argsort(all_num_inliers, descending=True).numpy()
    inds = inds[:, sorted_order]
    dists = dists[:, sorted_order]
    all_num_inliers = all_num_inliers[sorted_order]
    all_num_outliers = all_num_outliers[sorted_order]
    all_kptsA = all_kptsA[sorted_order]
    all_kptsB = all_kptsB[sorted_order]
    all_masks = all_masks[sorted_order]
    if ransac_mode == "homography":
        all_homographies = all_homographies[sorted_order]
    elif ransac_mode == "pnp":
        all_Rs = all_Rs[sorted_order]
        all_tvecs = all_tvecs[sorted_order]
        # all_covs = [all_covs[j] for j in sorted_order]

    # # bootstrap-based pose variance is only ever estimated for the winning
    # # (highest-inlier) candidate, not every candidate that went through PnP
    # bootstrap_pose_var = None
    # if ransac_mode == "pnp" and bootstrap_ransac is not None and best_dem is not None:
    #     print(f"RANSAC mode: {ransac_mode}, bootstrap_ransac: {bootstrap_ransac}, best_dem: {best_dem is not None}")
    #     _, _, _, bootstrap_pose_var = bootstrap_ransac(
    #         all_kptsA[0], all_kptsB[0], K, best_dem, best_geom
    #     )

    return dict(
        inds=inds,
        dists=dists,
        all_num_inliers=all_num_inliers.tolist(),
        all_homographies=all_homographies.tolist() if ransac_mode == "homography" else None,
        all_Rs=all_Rs.tolist() if ransac_mode == "pnp" else None,
        all_tvecs=all_tvecs.tolist() if ransac_mode == "pnp" else None,
        # all_pose_covs=all_covs if ransac_mode == "pnp" else None,
        # bootstrap_pose_var=bootstrap_pose_var,
        all_num_outliers=all_num_outliers.tolist(),
        qry_kpts=all_kptsA.tolist(),
        ref_kpts=all_kptsB.tolist(),
        inliers=all_masks.tolist(),
    )
