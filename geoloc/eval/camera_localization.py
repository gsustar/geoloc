import torch
import numpy as np
import cv2


def predict_qry_camera_position(
    ref_center_point,
    ref_original_shape,
    qry_matcher_shape,
    ref_matcher_shape,
    H,
    gsd_x=0.5,
    gsd_y=0.5,      
):
    if H is None:
        return None

    H = np.array(H, dtype=np.float64)

    qry_center = np.array([qry_matcher_shape[1] / 2, qry_matcher_shape[0] / 2, 1.0], dtype=np.float64)
    ref_center = np.array([ref_matcher_shape[1] / 2, ref_matcher_shape[0] / 2, 1.0], dtype=np.float64)

    projected_center = H @ qry_center
    if np.abs(projected_center[2]) < 1e-8:
        return ref_center_point
    projected_center = projected_center[:2] / projected_center[2]

    delta_center = projected_center - ref_center.squeeze()[:2]
    dx_pixels = delta_center[0]
    dy_pixels = delta_center[1]

    new_gsd_x = gsd_x * (ref_original_shape[1] / ref_matcher_shape[1])
    new_gsd_y = gsd_y * (ref_original_shape[0] / ref_matcher_shape[0])
    estimated_x = ref_center_point[0] + dx_pixels * new_gsd_x
    estimated_y = ref_center_point[1] - dy_pixels * new_gsd_y

    return np.array([estimated_x, estimated_y], dtype=np.float64)


def get_K_mors_gopro(W, H):
    base_w, base_h = 3840.0, 2160.0
    fx0, fy0, cx0, cy0 = 1831.2, 1831.2, 1920.0, 1080.0

    sx = W / base_w
    sy = H / base_h

    fx = fx0 * sx
    fy = fy0 * sy
    cx = cx0 * sx
    cy = cy0 * sy

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return K

def adjust_K_for_resize(K, original_w, original_h, new_w, new_h):
    sx = new_w / original_w
    sy = new_h / original_h
    K[0, 0] *= sx  # fx
    K[1, 1] *= sy  # fy
    K[0, 2] *= sx  # cx
    K[1, 2] *= sy  # cy
    return K

def adjust_K_for_centercrop(K, original_w, original_h, cropped_w, cropped_h):
    x_off = (original_w - cropped_w) / 2.0
    y_off = (original_h - cropped_h) / 2.0
    K[0, 2] -= x_off  # cx
    K[1, 2] -= y_off  # cy
    return K


def get_K_match_space(match_size, crop_w=1080, crop_h=1080):
    K = get_K_mors_gopro(1920, 1080)
    K = adjust_K_for_centercrop(K, 1920, 1080, crop_w, crop_h)
    K = adjust_K_for_resize(K, crop_w, crop_h, match_size, match_size)
    return K


def predict_qry_camera_position_pnp(R, tvec):
    # dem = ref["dem"]
    # rh, rw = dem.shape
    # minx, miny, maxx, maxy = ref["geometry"].bounds

    # matchedA = matchedA.squeeze()
    # matchedB = matchedB.squeeze()
    # if isinstance(matchedA, torch.Tensor):
    #     matchedA = matchedA.cpu().numpy()
    # if isinstance(matchedB, torch.Tensor):
    #     matchedB = matchedB.cpu().numpy()

    # matchedA = matchedA[matchedA[:, 0] >= 0]
    # matchedB = matchedB[matchedB[:, 0] >= 0]

    # matchedB_world = np.zeros((matchedB.shape[0], 3), dtype=np.float64)
    # matchedB_world[:, 0] = matchedB[:, 0] / rw * (maxx - minx) + minx
    # matchedB_world[:, 1] = (1 - matchedB[:, 1] / rh) * (maxy - miny) + miny
    # matchedB_world[:, 2] = dem[matchedB[:, 1].astype(int), matchedB[:, 0].astype(int)]

    # object_points = matchedB_world
    # image_points = matchedA
    # dist_coeffs = None

    # success, rvec, tvec, inliers = cv2.solvePnPRansac(
    #     object_points,
    #     image_points,
    #     K,
    #     dist_coeffs,
    #     iterationsCount=1000,
    #     reprojectionError=1.0,
    #     confidence=0.999,
    #     flags=cv2.SOLVEPNP_EPNP
    # )
    # R, _ = cv2.Rodrigues(rvec)
    if isinstance(R, list):
        R = np.array(R, dtype=np.float64)
    if isinstance(tvec, list):
        tvec = np.array(tvec, dtype=np.float64)

    camera_position_world = -R.T @ tvec

    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    yaw   = np.arctan2(R[1,0], R[0,0])
    pitch = np.arctan2(-R[2,0], sy)
    roll  = np.arctan2(R[2,1], R[2,2])

    yaw = np.degrees(yaw)
    pitch = np.degrees(pitch)
    roll = np.degrees(roll)

    return camera_position_world, (yaw, pitch, roll)