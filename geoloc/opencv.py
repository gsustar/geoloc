import cv2
import torch
import numpy as np


def refine_pose_lm(object_points, image_points, K, rvec, tvec):
    # Refine a PnP solution with Levenberg-Marquardt nonlinear optimization.
    obj = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.ascontiguousarray(K, dtype=np.float64)
    rvec = np.ascontiguousarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.ascontiguousarray(tvec, dtype=np.float64).reshape(3, 1)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    except cv2.error:
        pass
    return rvec, tvec

class HomographyEstimator:
    def __init__(self, reproj_threshold=1.0, maxIters=10000):
        self._find_homography = (lambda kptA, kptB:
            cv2.findHomography(
                kptA, kptB,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=reproj_threshold,
                confidence=0.999999,
                maxIters=maxIters
            )
        )

    def __call__(self, kptA, kptB):
        if isinstance(kptA, torch.Tensor):
            kptA = kptA.cpu().numpy()
        if isinstance(kptB, torch.Tensor):
            kptB = kptB.cpu().numpy()

        if kptA.shape[0] >= 4 and kptB.shape[0] >= 4:
            H, mask = self._find_homography(kptA, kptB)
        else:
            H = None
            mask = None

        H = torch.from_numpy(H).float() if H is not None else torch.eye(3)
        if mask is not None:
            mask = torch.from_numpy(mask).bool()
        else:
            mask = torch.zeros((kptA.shape[0], 1), dtype=torch.bool)

        return H, mask
    
class PnPSolver:
    def __init__(self, reproj_threshold=1.0, maxIters=1000):
        self._solvePnP = (lambda K, object_points, image_points:
            cv2.solvePnPRansac(
                object_points,
                image_points,
                K,
                distCoeffs=None,
                iterationsCount=maxIters,
                reprojectionError=reproj_threshold,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP
            )
        )

    def __call__(self, kptA, kptB, K, dem, geom):
        rh, rw = dem.shape
        minx, miny, maxx, maxy = geom.bounds

        matchedA = kptA.squeeze()
        matchedB = kptB.squeeze()
        if isinstance(matchedA, torch.Tensor):
            matchedA = matchedA.cpu().numpy()
        if isinstance(matchedB, torch.Tensor):
            matchedB = matchedB.cpu().numpy()

        if matchedA.ndim == 1:
            matchedA = matchedA[None, :]
        if matchedB.ndim == 1:
            matchedB = matchedB[None, :]
        num_input = matchedA.shape[0]
        valid = (matchedA[:, 0] >= 0) & (matchedB[:, 0] >= 0)
        matchedA = matchedA[valid]
        matchedB = matchedB[valid]

        matchedB_world = np.zeros((matchedB.shape[0], 3), dtype=np.float64)
        matchedB_world[:, 0] = matchedB[:, 0] / rw * (maxx - minx) + minx
        matchedB_world[:, 1] = (1 - matchedB[:, 1] / rh) * (maxy - miny) + miny
        row_idx = np.clip(matchedB[:, 1].astype(int), 0, rh - 1)
        col_idx = np.clip(matchedB[:, 0].astype(int), 0, rw - 1)
        matchedB_world[:, 2] = dem[row_idx, col_idx]

        centroid = matchedB_world.mean(axis=0) if matchedB_world.shape[0] > 0 else np.zeros(3)
        object_points = matchedB_world - centroid
        image_points = np.ascontiguousarray(matchedA, dtype=np.float64)
        if object_points.shape[0] >= 4 and image_points.shape[0] >= 4:
            success, rvec, tvec, inliers = self._solvePnP(K, object_points, image_points)
        else:
            success = False
            rvec = None
            tvec = None
            inliers = None

        if rvec is not None and tvec is not None and inliers is not None and success:
            # solvePnPRansac returns inlier INDICES, not a boolean mask
            inlier_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
            if inlier_idx.shape[0] >= 4:
                # rvec, tvec = refine_pose_lm(object_points, image_points, K, rvec, tvec)
                rvec, tvec = refine_pose_lm(object_points[inlier_idx], image_points[inlier_idx], K, rvec, tvec)
            R, _ = cv2.Rodrigues(rvec)
            # undo the centering: t_world = t_centered - R m
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1) - R @ centroid.reshape(3, 1)
            R = torch.from_numpy(R).float()
            tvec = torch.from_numpy(tvec).float()
            # scatter inlier indices back to a boolean mask over the input keypoint rows
            mask_valid = np.zeros(matchedA.shape[0], dtype=bool)
            mask_valid[inlier_idx] = True
            mask = np.zeros(num_input, dtype=bool)
            mask[valid] = mask_valid
            inliers = torch.from_numpy(mask).unsqueeze(1)
        else:
            R = torch.eye(3)
            tvec = torch.zeros((3, 1))
            inliers = torch.zeros((num_input, 1), dtype=torch.bool)

        return R, tvec, inliers
