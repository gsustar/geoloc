from __future__ import annotations

import torch
import numpy as np


def inv_pose(pose_matrix: np.ndarray) -> np.ndarray | torch.Tensor:
    """
    Invert a pose matrix.

    Args:
        pose_matrix: Pose matrix to invert.

    Returns:
        Inverted pose matrix.
    """
    is_numpy = isinstance(pose_matrix, np.ndarray)
    if is_numpy:
        pose_matrix = torch.from_numpy(pose_matrix)
    R, t = decompose_pose(pose_matrix)
    R_inv = R.transpose(-2, -1)
    t_inv = -R_inv @ t.unsqueeze(-1)
    if is_numpy:
        return compose_pose(R_inv, t_inv).numpy()
    return compose_pose(R_inv, t_inv)


def decompose_pose(pose_matrix: np.ndarray) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Decompose a pose matrix into rotation and translation components.

    Args:
        pose_matrix: Pose matrix to decompose.

    Returns:
        A tuple containing the rotation matrix and translation vector.
    """
    is_numpy = isinstance(pose_matrix, np.ndarray)
    if is_numpy:
        pose_matrix = torch.from_numpy(pose_matrix)
    # should operate on batchified and non-batchified poses
    if pose_matrix.ndimension() == 3:
        t = pose_matrix[:, :3, 3]
        R = pose_matrix[:, :3, :3]
    else:
        t = pose_matrix[:3, 3]
        R = pose_matrix[:3, :3]
    if is_numpy:
        return R.numpy(), t.numpy()
    return R, t


def compose_pose(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compose a pose matrix from rotation and translation components.

    Args:
        R: Rotation matrix.
        t: Translation vector.

    Returns:
        Composed pose matrix.
    """
    if R.ndimension() == 3:
        t = t.view(-1, 3, 1)
        return torch.cat([R, t], dim=2)
    return torch.cat([R, t.view(3, 1)], dim=1)


def compute_raster_intrinsics_extrinsics(scale: tuple[float, float] | np.ndarray,
                                         offset: tuple[float, float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the raster intrinsics and extrinsics from the scale and offset.
    """
    sx, sy = scale
    ox, oy = offset[:2]
    pose_world2dop = np.array([[1, 0, 0, -ox], [0, 1, 0, -oy], [0, 0, 0, 1]], dtype=np.float32)
    intrinsics_dop = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float32)
    return pose_world2dop, intrinsics_dop