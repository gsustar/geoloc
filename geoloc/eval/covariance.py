import cv2
import numpy as np


def skew(v):
    # Returns the skew-symmetric matrix of a 3-vector v
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def so3_right_jacobian(rvec):
    # Closed form for the right Jacobian of SO(3) - https://arxiv.org/pdf/1812.01537#page=15
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(r)
    S = skew(r)
    if theta < 1e-7:
        return np.eye(3) - 0.5 * S + (1.0 / 6.0) * (S @ S)
    t2 = theta * theta
    return (
        np.eye(3)
        - ((1.0 - np.cos(theta)) / t2) * S
        + ((theta - np.sin(theta)) / (t2 * theta)) * (S @ S)
    )


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


def pairwise_distance_ratios(object_points, image_points, n_samples=None, rng=0):
    """For every pair (i, j) of inlier correspondences, the ratio of the
    object-space distance to the image-space distance:

        ratio_ij = ||obj_i - obj_j|| / ||img_i - img_j||   (ref-px / image-px)

    object_points is (N, 3) in the reference orthophoto frame (x, y in ortho
    pixels; z is DEM elevation, see opencv.py), image_points is (N, 2) in drone
    image pixels; row i of each is the same correspondence.

    If n_samples is given and smaller than the number of pairs C(N,2), a random
    subset of that many pairs is used instead of all of them (rng: an int seed
    or np.random.Generator, defaulting to a fixed seed for reproducibility).

    Returns (i, j, obj_dist, img_dist, ratio), each a length-(#pairs) array over
    the sampled upper-triangular pairs. Pairs with coincident image points yield
    inf.
    """
    obj = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 2)

    i, j = np.triu_indices(obj.shape[0], k=1)
    if n_samples is not None and n_samples < i.size:
        sel = np.random.default_rng(rng).choice(i.size, size=n_samples, replace=False)
        i, j = i[sel], j[sel]
    obj_dist = np.linalg.norm(obj[i] - obj[j], axis=1)
    img_dist = np.linalg.norm(img[i] - img[j], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = obj_dist / img_dist
    return i, j, obj_dist, img_dist, ratio


def mean_distance_ratio(object_points, image_points, n_samples=None, rng=0):
    """Mean object/image distance ratio (reference px per drone-image px) over
    the inlier pairs, averaged across n_samples randomly drawn pairs (or all
    C(N,2) pairs if n_samples is None). Non-finite pairs (coincident image
    points) are dropped. Returns nan if no finite pair is available."""
    _, _, _, _, ratio = pairwise_distance_ratios(object_points, image_points, n_samples, rng)
    ratio = ratio[np.isfinite(ratio)]
    return float(np.mean(ratio)) if ratio.size else float("nan")


def pnp_pose_covariance(object_points, image_points, K, rvec, tvec, sigma_floor_px=0.5, scale_samples=1000, gsd_m_per_px=0.5):
    """First-order (Gauss-Newton) covariance of a PnP solution from the
    reprojection residuals: Sigma_(rvec,tvec) = sigma^2 (J^T J)^-1, then
    propagated to the camera pose in world coordinates.

    object_points/image_points must be the INLIER correspondences the pose was
    refined on. object_points are in the reference orthophoto frame (x, y in
    ortho pixels; z is DEM elevation - see opencv.py), image_points in drone
    image pixels. NOTE: because x, y are pixels, the position covariance below
    is in ortho pixels^2, NOT meters^2 (z carries the DEM's own units).

    gsd_m_per_px is the orthophoto ground sampling distance (default 0.5, i.e.
    1 ortho pixel = 0.5 m); it is used to also report the position covariance in
    metres via the *_m keys below.

    sigma^2 is estimated from the residuals as RSS / (2N - 6) and floored at
    sigma_floor_px^2 (RANSAC truncates inlier residuals at the reprojection
    threshold, which biases the estimate down).

    Returns a dict (plain python types, JSON-serializable) or None if the
    problem is under-constrained:
      cov_pose     6x6, order [x, y, z, rot_x, rot_y, rot_z]:
                   camera center C = -R^T t in world axes (ortho px for x, y;
                   DEM units for z) and a world-frame small-angle perturbation
                   of R_wc = R^T (rad)
      cov_pose_m   cov_pose with x, y rescaled to metres by gsd_m_per_px
                   (z assumed already metres); this is the metric pose covariance
      cov_rt       6x6 in the solver's (rvec, tvec) parametrization; the tvec
                   block inherits the object-point units (ortho px^2, not m^2)
      sigma_px     floored residual std used to scale the covariance
      rmse_px      raw per-coordinate RMS reprojection error of the inliers
      dof          2N - 6
      cond         condition number of J^T J (large => degenerate geometry)
      pos_std      [sx, sy, sz] in ortho pixels (world axes; z in DEM units)
      pos_std_m    [sx, sy, sz] in metres (x, y = pos_std * gsd_m_per_px; z as-is)
      gsd_m_per_px orthophoto ground sampling distance used for the conversion
      scale_refpx_per_imgpx  mean ortho-px per drone-image-px over inlier pairs
      scale_m_per_imgpx      that ratio in metres/drone-px (scale * gsd_m_per_px)
      rot_std_deg  [rx, ry, rz] in degrees (world axes; index 2 ~ yaw)
    """
    obj = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 2)
    K = np.ascontiguousarray(K, dtype=np.float64)
    rvec = np.ascontiguousarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.ascontiguousarray(tvec, dtype=np.float64).reshape(3, 1)

    n = obj.shape[0]
    dof = 2 * n - 6
    if dof < 2:
        return None

    # Center the world frame on the object-point centroid. With absolute
    # georeferenced coordinates the rvec Jacobian columns scale with the
    # distance to the world origin and J^T J is numerically singular.
    # cov(C) is invariant to the shift (C' = C - m, t' = t + R m).
    R_cw, _ = cv2.Rodrigues(rvec)
    m = obj.mean(axis=0)

    obj = obj - m
    tvec = tvec + R_cw @ m.reshape(3, 1)

    proj, jac = cv2.projectPoints(obj, rvec, tvec, K, None)
    residuals = (proj.reshape(-1, 2) - img).reshape(-1)
    # projectPoints jacobian columns: [d/drvec (3), d/dtvec (3), d/df, d/dc, d/ddist]
    J = np.asarray(jac, dtype=np.float64)[:, :6]

    rss = float(residuals @ residuals)
    rmse_px = float(np.sqrt(rss / (2 * n)))
    # sigma_px = max(np.sqrt(rss / dof), float(sigma_floor_px))
    sigma_px = np.sqrt(rss / dof)

    JtJ = J.T @ J
    eigvals = np.linalg.eigvalsh(JtJ)
    if eigvals[0] <= 1e-12 * max(eigvals[-1], 1.0):
        return None
    cond = float(eigvals[-1] / eigvals[0])
    cov_rt = (sigma_px ** 2) * np.linalg.inv(JtJ)
    cov_pose = np.zeros((6, 6), dtype=np.float64)
    
    # Propagate (drvec, dtvec) -> (dC, dtheta_world).
    # R_cw = R(rvec) maps world->camera; C = -R_cw^T t (here in the centered
    # frame; dC is identical to the uncentered one).
    # With R(rvec + dr) = R_cw exp((Jr dr)^):
    #   dC     =  [C]x Jr dr - R_cw^T dt
    #   dtheta = -Jr dr        (left perturbation of R_wc: exp(dtheta^) R_wc)
    C = (-R_cw.T @ tvec).reshape(3)
    Jr = so3_right_jacobian(rvec)
    T = np.zeros((6, 6), dtype=np.float64)
    T[:3, :3] = skew(C) @ Jr
    T[:3, 3:] = -R_cw.T
    T[3:, :3] = -Jr
    cov_pose = T @ cov_rt @ T.T

    # Mean object/image distance ratio (reference px per drone-image px) over a
    # sample of inlier pairs. Centering obj above does not affect pairwise dist.
    scale_refpx_per_imgpx = mean_distance_ratio(obj, img, n_samples=scale_samples)

    pos_std = np.sqrt(np.clip(np.diag(cov_pose[:3, :3]), 0.0, None))
    rot_std = np.sqrt(np.clip(np.diag(cov_pose[3:, 3:]), 0.0, None))

    # Convert the position block from the ortho-pixel frame to metres. 1 ortho
    # pixel = gsd_m_per_px metres, so x, y scale by the GSD; z is already in
    # DEM metres. Change of variables on the covariance: cov_m = S cov S^T with
    # S = diag([gsd, gsd, 1, 1, 1, 1]) (rotation block untouched).
    S = np.diag([gsd_m_per_px, gsd_m_per_px, 1.0, 1.0, 1.0, 1.0])
    cov_pose_m = S @ cov_pose @ S.T
    pos_std_m = np.sqrt(np.clip(np.diag(cov_pose_m[:3, :3]), 0.0, None))
    # Ground sampling distance of the drone image (metres of ground per drone px).
    scale_m_per_imgpx = scale_refpx_per_imgpx * gsd_m_per_px

    return dict(
        scale_refpx_per_imgpx=scale_refpx_per_imgpx,
        scale_m_per_imgpx=scale_m_per_imgpx,
        gsd_m_per_px=float(gsd_m_per_px),
        cov_pose=cov_pose.tolist(),
        cov_pose_m=cov_pose_m.tolist(),
        cov_rt=cov_rt.tolist(),
        sigma_px=float(sigma_px),
        rmse_px=rmse_px,
        dof=int(dof),
        cond=cond,
        pos_std=pos_std.tolist(),
        pos_std_m=pos_std_m.tolist(),
        rot_std_deg=np.degrees(rot_std).tolist(),
    )
