import os
import cv2
import time
import torch
import einops
import pickle

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from scipy.spatial import Delaunay

from ..utils import DEBUG
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

PATH_TO_PREFITTED_CENTERS = (
    "/storage/datasets/AerialLoc/Drone2Sat/ckpts/anyloc_aerial_c_centers/c_centers.pt"
)


class SegVLAD(nn.Module):
    def __init__(
        self,
        segmentor,
        backbone,
        vlad,
        pca=None,
    ):
        super().__init__()
        self.segmentor = segmentor
        self.backbone = backbone
        self.vlad = vlad
        self.pca = pca

    # def _load_prefitted_vlad(self):
    # 	vlad_pre = torch.hub.load("AnyLoc/DINO", "get_vlad_model",
    # 				backbone="DINOv2", domain="aerial").vlad
    # 	assert (self.vlad.mode == "cosine" and
    # 	  		self.vlad.vlad_mode == "hard" and
    # 			self.vlad.desc_dim == vlad_pre.desc_dim and
    # 			self.vlad.num_clusters == vlad_pre.num_clusters), (
    # 	"Pre-fitted VLAD model parameters do not match the configuration")
    # 	vlad_pre_c_centers = vlad_pre.c_centers.cpu().clone()
    # 	self.vlad.fit(None, c_centers=vlad_pre_c_centers)
    # 	del vlad_pre
    # 	return self.vlad

    def _load_prefitted_vlad(self):
        assert (
            self.vlad.mode == "cosine"
            and self.vlad.vlad_mode == "hard"
            and self.vlad.desc_dim == 1536
            and self.vlad.num_clusters == 32
        ), "Pre-fitted VLAD model parameters do not match the configuration"
        vlad_pre_c_centers = torch.load(PATH_TO_PREFITTED_CENTERS)
        self.vlad.fit(None, c_centers=vlad_pre_c_centers)
        return self.vlad

    def fit_and_generate(
        self,
        ref_image_dataloader,
        savedir,
        vlad_fitstep=1,
        use_prefitted_vlad=False,
        device="cpu",
    ):
        print("Extracting features...")
        features = []
        for i, batch in enumerate(tqdm(ref_image_dataloader)):
            if DEBUG > 1 and i > 20:
                break
            if i % vlad_fitstep != 0:
                continue
            image = batch["image"]
            _, masks = self.segmentor(image)
            # image = batch["image"].to(device)
            x = self.backbone(image.to(device))
            B, C, _, _ = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
            features.append(x.detach())
            if self.pca is None and use_prefitted_vlad:
                break  # Early exit in case of pre-fitted VLAD and no PCA, only need features for asserting vector dimension
        features = torch.vstack(features)

        if use_prefitted_vlad:
            self.vlad = self._load_prefitted_vlad()
            features = self.vlad.generate_multi(features)
            self.vlad.save()  # Manually save `c_centers` (usually this happens during `fit`)
        else:
            print("Fitting VLAD features...")
            assert self.vlad.cache_dir is not None
            assert self.vlad.cache_dir == savedir
            features = self.vlad.fit_and_generate(features)

        # TODO: test if PCA works
        if self.pca is not None:
            print("Fitting PCA...")
            features = self.pca.fit_transform(features.cpu())
            self.save_pca(savedir)
            print(f"PCA components: {self.pca.components_.shape}")
        return features

    def load_vlad(self, loaddir):
        assert self.vlad.cache_dir is not None
        assert self.vlad.cache_dir == loaddir
        assert os.path.exists(os.path.join(loaddir, "c_centers.pt"))
        self.vlad.fit(None)

    def load_pca(self, loaddir):
        pca_loadpath = os.path.join(loaddir, "pca.pkl")
        if os.path.exists(pca_loadpath):
            print("Loading PCA from file...")
            with open(pca_loadpath, "rb") as f:
                self.pca = pickle.load(f)

    def save_pca(self, savedir):
        if self.pca is not None:
            with open(os.path.join(savedir, "pca.pkl"), "wb") as f:
                pickle.dump(self.pca, f)

    def forward(self, x, return_residuals=False):
        x = self.backbone(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        if return_residuals:
            qry_residuals = self.vlad.generate_multi_res_vec(x)
        x = self.vlad.generate_multi(x)
        x = (
            torch.from_numpy(self.pca.transform(x)).float()
            if self.pca is not None
            else x
        )

        if return_residuals:
            return x, qry_residuals
        return x


############################################################################
###   Helper functions from https://github.com/AnyLoc/Revisit-Anything   ###
############################################################################


def seg_vlad_gpu_single(
    ind, idx, dino_desc, segMask, c_centers, cfg, desc_dim=1536, adj_mat=None
):
    dh = cfg["desired_height"] // 14
    dw = cfg["desired_width"] // 14

    total_elements = dino_desc.shape[2] * dino_desc.shape[3]
    dino_desc = dino_desc.reshape(1, desc_dim, total_elements).to("cuda")

    dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)

    # IMPORTANT: could be wrong here
    mask = torch.from_numpy(np.array(segMask)).to("cuda")
    mask = (
        torch.nn.functional.interpolate(
            mask.float().unsqueeze(0),
            [cfg["desired_height"], cfg["desired_width"]],
            mode="nearest",
        )
        .squeeze()
        .bool()
        .reshape(len(segMask), -1)
    )
    mask_idx = torch.zeros((len(segMask), dh * dw), device="cuda").bool()
    mask_ind_flat = torch.argwhere(mask)
    mask_idx[mask_ind_flat[:, 0], ind[mask_ind_flat[:, 1]]] = True

    reg_feat_per = dino_desc_norm.permute(0, 2, 1)
    if adj_mat is not None:
        gd, execution_time = vlad_single(
            reg_feat_per.squeeze(),
            c_centers.to("cuda"),
            mask_idx,
            adj_mat.to("cuda"),
        )
    else:
        gd, execution_time = vlad_single(
            reg_feat_per.squeeze(), c_centers.to("cuda"), mask_idx
        )

    return gd.cpu(), execution_time


def vlad_single(query_descs, c_centers, masks, adj_mat=None):
    num_clusters = 32
    c_centers_norm = F.normalize(c_centers, dim=1).to("cuda")
    labels = torch.argmax(query_descs @ c_centers_norm.T, dim=1)
    residuals = query_descs - c_centers[labels]
    if adj_mat is not None:
        n_vlad, execution_time = vlad_matmuls_per_cluster(
            num_clusters,
            masks.double(),
            residuals.double(),
            labels,
            adjMat=adj_mat.double(),
        )
    else:
        n_vlad, execution_time = vlad_matmuls_per_cluster(
            num_clusters, masks.double(), residuals.double(), labels
        )
    return n_vlad, execution_time


def vlad_matmuls_per_cluster(
    num_c, masks, res, clus_labels, adjMat=None, device="cuda"
):
    """
    Expects input tensors to be cuda and float/double
    """
    start_time = time.time()

    vlads = []
    num_m = len(masks)

    if adjMat is None:
        adjMat = torch.eye(num_m, dtype=masks.dtype, device=masks.device)

    for li in range(num_c):
        inds_li = torch.where(clus_labels == li)[0].to(device)
        masks_nbrAgg = adjMat @ masks[:, inds_li]
        vlad = masks_nbrAgg.bool().to(masks.dtype) @ res[inds_li, :]
        vlad = F.normalize(vlad, dim=1)
        vlads.append(vlad)
    vlads = torch.stack(vlads).permute(1, 0, 2).reshape(len(masks), -1)
    vlads = F.normalize(vlads, dim=1)

    end_time = time.time()
    execution_time = end_time - start_time
    return vlads, execution_time


def nbrMasksAGGFastSingle(masks_seg, order=1):
    mask_cords = np.array(
        [np.array(np.nonzero(mask_seg)).mean(1)[::-1] for mask_seg in masks_seg]
    )
    adj_mat = torch.zeros((len(mask_cords), len(mask_cords)))

    if len(mask_cords) > 3:
        tri = Delaunay(mask_cords)
        for v in range(len(mask_cords)):
            nbrsList = getNbrsDelaunay(tri, v)
            nbrsList = np.unique([[v, v]] + nbrsList)
            adj_mat[v][nbrsList] = 1

        adj_mat_power = adj_mat.clone()
        for _ in range(order - 1):  # Multiply original matrix order-1 times
            adj_mat_power = adj_mat_power @ adj_mat
        adj_mat = adj_mat_power.bool()

    else:
        nbr_list = (
            [0, 1] if len(mask_cords) > 1 else [0]
        )  # Adjusting for cases with less than 2 masks
        for v in range(len(mask_cords)):
            adj_mat[v][nbr_list] = 1
        adj_mat = adj_mat.bool()

    return adj_mat


def getNbrsDelaunay(tri, v):
    indptr, indices = tri.vertex_neighbor_vertices
    v_nbrs = indices[indptr[v] : indptr[v + 1]]
    v_nbrs = [[v, u] for u in v_nbrs]
    return v_nbrs


def getIdxSingleFast(img_idx, masks_seg, minArea=400, returnMask=True):
    imInds = []
    regIndsIm = []
    segmask = []
    count = 0
    for mask in masks_seg:
        if returnMask:
            segmask.append(mask)
        regIndsIm.append(count)
        imInds.append(img_idx)
        count += 1

    return np.array(imInds), regIndsIm, segmask


def process_single_SAM(cfg, img, models):
    mask_generator = models
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cfg["resize"]:
        img_p = cv2.resize(img, (cfg["desired_width"], cfg["desired_height"]))

    else:
        img_p = img
    masks = mask_generator.generate(img_p)
    return img_p, masks


def loadSAM(sam_checkpoint, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def apply_pca_transform_from_pkl(data_tensor, pca_model_path):
    """
    Loads a PCA model from a pickle file and applies the transform to the given data tensor.

    Parameters:
    - data_tensor: A PyTorch tensor containing the data to be transformed.
    - pca_model_path: Path to the pickle file containing the fitted PCA model.

    Returns:
    - A PyTorch tensor containing the transformed data.
    """
    # Ensure data is on CPU and converted to a NumPy array for PCA transformation
    data_np = data_tensor.cpu().numpy()

    # Load the fitted PCA model from disk
    with open(pca_model_path, "rb") as file:
        pca = pickle.load(file)

    # Apply the PCA transform to the data
    transformed_data_np = pca.transform(data_np)

    # Convert the transformed data back to a PyTorch tensor
    transformed_data_tensor = torch.from_numpy(transformed_data_np)

    return transformed_data_tensor
