import cv2
import torch
import numpy as np


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
        matchedA = matchedA[matchedA[:, 0] >= 0]
        matchedB = matchedB[matchedB[:, 0] >= 0]

        matchedB_world = np.zeros((matchedB.shape[0], 3), dtype=np.float64)
        matchedB_world[:, 0] = matchedB[:, 0] / rw * (maxx - minx) + minx
        matchedB_world[:, 1] = (1 - matchedB[:, 1] / rh) * (maxy - miny) + miny
        matchedB_world[:, 2] = dem[matchedB[:, 1].astype(int), matchedB[:, 0].astype(int)]

        object_points = matchedB_world
        image_points = matchedA
        if object_points.shape[0] >= 4 and image_points.shape[0] >= 4:
            success, rvec, tvec, inliers = self._solvePnP(K, object_points, image_points)
        else:
            rvec = None
            tvec = None
            inliers = None
        
        if rvec is not None and tvec is not None and inliers is not None and success:
            R, _ = cv2.Rodrigues(rvec)
            R = torch.from_numpy(R).float()
            tvec = torch.from_numpy(tvec).float()
            inliers = torch.from_numpy(inliers).bool()
        else:
            R = torch.eye(3)
            tvec = torch.zeros((3, 1))
            inliers = torch.zeros((matchedA.shape[0], 1), dtype=torch.bool)

        return R, tvec, inliers
