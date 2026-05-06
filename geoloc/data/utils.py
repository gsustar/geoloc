import os
import cv2
import json
import torch
import torchvision
from typing import List
from natsort import natsorted
import numpy as np

import torchvision.transforms.functional as F

from torch.utils.data._utils.collate import default_collate
from torch.utils.data import default_collate


def is_image(path_or_filename: str) -> bool:
    """Check if path is an image file"""
    return path_or_filename.lower().endswith((".jpg", ".png", ".jpeg", ".tif"))


def get_sorted_imgpaths(directory: str) -> List[str]:
    """Get sorted list of absolute image paths in directory"""
    return natsorted(
        [
            os.path.join(directory, img)
            for img in os.listdir(directory)
            if is_image(os.path.join(directory, img))
        ]
    )


def load_image(path: str) -> torch.Tensor:
    """Load an image as a float32 torch.Tensor"""
    img = torchvision.io.read_image(path).to(torch.float32)
    if img.shape[0] > 3:
        img = img[:3]
    return img


def pad_to_size(img: torch.Tensor, target_size: int) -> torch.Tensor:
    """Zero-pad image to target size"""
    _, h, w = img.shape
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    ptile = F.pad(img, (0, 0, pad_w, pad_h), fill=0)
    return ptile


def video_to_trajectory(
    video_path,
    sampling_rate_frames=10,
    sampling_rate_seconds=None,
    center_crop=None,
    skip_first_n_samples=0,
):
    """
    Convert a video to a trajectory of images by saving frames as images.

    Args:
            video_path (str): Path to the input video file. The directory of this video should also contain a json file with metadata.
            sampling_rate_frames (int): Save every 'sampling_rate' frame. default is 10.
            sampling_rate_seconds (int): Save every 'sampling_rate_seconds' seconds. if provided, it will override sampling_rate_frames.
            center_crop (int): If provided, crop the center of the frame to this size.
    """

    assert os.path.exists(video_path), f"Video file does not exist: {video_path}"

    metadata_path = os.path.splitext(video_path)[0] + ".json"
    assert os.path.exists(
        metadata_path
    ), f"Metadata file does not exist: {metadata_path}"

    video_dir = os.path.dirname(video_path)
    output_dir = os.path.join(video_dir, "queries")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Determine frame interval
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = sampling_rate_frames
    if sampling_rate_seconds is not None:
        frame_interval = max(int(fps * sampling_rate_seconds), 1)

    # Open metadata file
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    curr_sec = 0
    frame_idx = 0
    saved_idx = 0
    skip_counter = 0
    frame_metadata = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if skip_counter < skip_first_n_samples:
            skip_counter += 1
            frame_idx += 1
            continue

        if frame_idx % frame_interval == 0:
            curr_sec = int(frame_idx / fps)

            if center_crop is not None:
                fh, fw, _ = frame.shape
                cx, cy = fw // 2, fh // 2  # center
                h = w = center_crop
                x = cx - w / 2  # left
                y = cy - h / 2  # top
                frame = frame[int(y) : int(y + h), int(x) : int(x + w)]

            frame_filename = os.path.join(output_dir, f"{saved_idx:05d}.jpg")
            frame_metadata[frame_filename] = {
                "lon": metadata[curr_sec]["content"]["GPS"]["longitude"],
                "lat": metadata[curr_sec]["content"]["GPS"]["latitude"],
            }
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {saved_idx} at second {curr_sec}")
            saved_idx += 1

        frame_idx += 1

    # Save metadata to json file
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(frame_metadata, f, indent=2)

    cap.release()
    print(f"Saved {saved_idx} frames to {output_dir}")


def collate_with_geometry(batch):
    """
    Collate function that uses PyTorch's default_collate for everything,
    except keeps 'geometry' entries as a list of Shapely geometries.
    """
    elem = batch[0]
    if isinstance(elem, dict):
        collated = {}
        for key in elem:
            if key == "geometry":
                collated[key] = [sample[key] for sample in batch]
            elif key == "gt_pos" and all([isinstance(x[key], np.ndarray) for x in batch]):
                maxn = max([sample[key].shape[0] for sample in batch])
                padded = []
                for sample in batch:
                    pad_size = maxn - sample[key].shape[0]
                    padded_tensor = torch.from_numpy(
                        np.concatenate([sample[key], sample[key][-1].repeat(max(0, pad_size))], axis=0)
                    )
                    padded.append(padded_tensor)
                collated[key] = torch.stack(padded, dim=0)
            else:
                collated[key] = default_collate([sample[key] for sample in batch])
        return collated
    else:
        return default_collate(batch)


def get_valid_center(array: np.ndarray, mask: np.ndarray):
    """
    Get the valid element closest to the image center.
    
    Args:
        array: Array of shape [H, W, ...] or [H, W]
        mask: Boolean mask of shape [H, W], True = valid
        
    Returns:
        value: The array element at closest valid position
        coords: (y, x) tuple of the position
        distance: Euclidean distance from true center
    """
    if not torch.any(mask).item():
        raise ValueError("Mask has no valid pixels")
    
    h, w = mask.shape
    center_y, center_x = h / 2 - 0.5, w / 2 - 0.5  # Exact center (can use h//2, w//2 for pixel center)
    
    # Get coordinates of all valid pixels [N, 2]
    valid_coords = torch.argwhere(mask)  # Returns [[y1, x1], [y2, x2], ...]
    
    # Compute squared distance to center for all valid pixels
    distances_sq = torch.sum((valid_coords - torch.tensor([center_y, center_x])) ** 2, axis=1)
    
    # Find closest
    closest_idx = torch.argmin(distances_sq)
    closest_y, closest_x = valid_coords[closest_idx]
    distance = torch.sqrt(distances_sq[closest_idx]).item()
    
    return array[closest_y, closest_x]#, (closest_y, closest_x), distance


def build_spatial_coords(ref_image_dataset, target_size: int):
    def _coords_from_all_windows(all_windows):
        bounds = all_windows.geometry.bounds
        center_x = ((bounds["minx"] + bounds["maxx"]) / 2.0).to_numpy(dtype=np.float64)
        center_y = ((bounds["miny"] + bounds["maxy"]) / 2.0).to_numpy(dtype=np.float64)
        return np.stack([center_y, center_x], axis=1)

    if hasattr(ref_image_dataset, "all_windows"):
        coords = _coords_from_all_windows(ref_image_dataset.all_windows)
    elif hasattr(ref_image_dataset, "_datasets"):
        all_coords = []
        for ds in ref_image_dataset._datasets:
            if not hasattr(ds, "all_windows"):
                raise ValueError("Expected GURS sub-dataset with `all_windows` attribute.")
            all_coords.append(_coords_from_all_windows(ds.all_windows))
        coords = np.concatenate(all_coords, axis=0) if len(all_coords) > 0 else np.empty((0, 2), dtype=np.float64)
    else:
        raise ValueError("Expected GURS reference dataset exposing `all_windows` (or `_datasets` with `all_windows`).")

    if coords.shape[0] == target_size:
        return coords

    if coords.shape[0] == 0 or target_size % coords.shape[0] != 0:
        raise ValueError(
            f"Cannot align spatial coordinates length ({coords.shape[0]}) to vdb size ({target_size})."
        )

    return np.repeat(coords, target_size // coords.shape[0], axis=0)