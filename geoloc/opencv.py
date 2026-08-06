import cv2
import torch
import numpy as np

from geoloc.eval.covariance import refine_pose_lm, pnp_pose_covariance


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
    def __init__(self, reproj_threshold=1.0, maxIters=1000, refine_lm=True, sigma_floor_px=0.5):
        self.refine_lm = refine_lm
        self.sigma_floor_px = sigma_floor_px
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
        matchedB_world[:, 2] = dem[matchedB[:, 1].astype(int), matchedB[:, 0].astype(int)]

        # Solve in a world frame centered on the object points: with absolute
        # georeferenced coordinates the LM refinement normal equations are
        # numerically singular. The returned tvec is shifted back below.
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

        cov_info = None
        if rvec is not None and tvec is not None and inliers is not None and success:
            # solvePnPRansac returns inlier INDICES, not a boolean mask
            inlier_idx = np.asarray(inliers, dtype=np.int64).reshape(-1)
            obj_in = object_points[inlier_idx]
            img_in = image_points[inlier_idx]
            if self.refine_lm and inlier_idx.shape[0] >= 4:
                rvec, tvec = refine_pose_lm(obj_in, img_in, K, rvec, tvec)
            cov_info = pnp_pose_covariance(
                obj_in, img_in, K, rvec, tvec, sigma_floor_px=self.sigma_floor_px
            )
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

        return R, tvec, inliers, cov_info
