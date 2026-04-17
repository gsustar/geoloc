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
    # H4 = np.eye(4, dtype=np.float64)
    # H4[:3, :3] = H

    # qry_center = np.array([[[qry_matcher_shape[1] / 2, qry_matcher_shape[0] / 2, 1.0]]], dtype=np.float64)
    # ref_center = np.array([[[ref_matcher_shape[1] / 2, ref_matcher_shape[0] / 2, 1.0]]], dtype=np.float64)
    qry_center = np.array([qry_matcher_shape[1] / 2, qry_matcher_shape[0] / 2, 1.0], dtype=np.float64)
    ref_center = np.array([ref_matcher_shape[1] / 2, ref_matcher_shape[0] / 2, 1.0], dtype=np.float64)

    # projected_center = cv2.perspectiveTransform(qry_center, H4).squeeze()
    # projected_center = projected_center[:2] / projected_center[2]
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