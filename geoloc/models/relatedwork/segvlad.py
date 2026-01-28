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

from ...utils import DEBUG
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from .anyloclike import AnyLocLikeModel
from ..aggregators.vlad import VLAD

# PATH_TO_PREFITTED_CENTERS = (
#     "/storage/datasets/AerialLoc/Drone2Sat/ckpts/anyloc_aerial_c_centers/c_centers.pt"
# )

class SegVLAD(AnyLocLikeModel):
    def __init__(
        self,
        desired_height: int,
        desired_width: int,
        sam_checkpoint: str,
        sam_model_type: str = "vit_h",
        vlad_checkpoint: str = None,
        order=3,
        sam_resize=True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.aggregator, VLAD)
        self.segmentor = loadSAM(
            sam_checkpoint=sam_checkpoint,
            model_type=sam_model_type, device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.vlad = self.aggregator
        self.sam_resize = sam_resize
        self.order = order

        if vlad_checkpoint is not None:
            self.vlad.load_c_centers(vlad_checkpoint)

        self.desired_height = desired_height
        self.desired_width = desired_width
        dh = desired_height // 14
        dw = desired_width // 14

        idx_matrix = np.empty((desired_height, desired_width, 2)).astype('int32')
        for i in range(desired_height):
            for j in range(desired_width):
                idx_matrix[i, j] = np.array([np.clip(i//14, 0, dh-1), np.clip(j//14, 0, dw-1)])
        ind_matrix = np.ravel_multi_index(idx_matrix.reshape(-1, 2).T, (dh, dw))
        self.register_buffer("ind_matrix", torch.tensor(ind_matrix))


    def extract_vlad_features(self, x: torch.Tensor, idx: int):
        bs = x.shape[0]
        assert bs == 1, "Batch size greater than 1 not supported in SegVLAD"
        img = x.clone()[0]
        img = (img * 255).cpu().numpy().astype(np.uint8).transpose(1,2,0)
        with torch.no_grad():
            masks = process_single_SAM(img, self.segmentor, resize=True)
        masks_seg = [mask_data["segmentation"] for mask_data in masks]

        imInds1_ind, regInds1_ind, segMask1_ind = getIdxSingleFast(idx, masks_seg)
        # imInds1 = np.concatenate((imInds1, imInds1_ind))
        if self.order: 
            adjMat1_ind = nbrMasksAGGFastSingle(masks_seg, self.order).to(self.device)
        else:
            adjMat1_ind = None

        with torch.no_grad():
            # dino_desc = self.backbone(img.unsqueeze(0))
            dino_desc = self.backbone(x)
        gd, _ = seg_vlad_gpu_single(self.ind_matrix, dino_desc, segMask1_ind, self.vlad.c_centers, self.desired_height, self.desired_width, desc_dim=1536, adj_mat=adjMat1_ind)#, device=self.device)

        return dict(
            out=gd,
            # imInds1=imInds1,
            imInds1_ind=imInds1_ind,
            regInds1_ind=regInds1_ind,
            segMask1_ind=segMask1_ind,
        )

    def forward(self, x: torch.Tensor, idx: int):
        # bs = x.shape[0]
        # assert bs == 1, "Batch size greater than 1 not supported in SegVLAD"
        # img = x.clone()[0]
        # img = (img * 255).cpu().numpy().astype(np.uint8).transpose(1,2,0)
        # with torch.no_grad():
        #     masks = process_single_SAM(img, self.segmentor, resize=True)
        # masks_seg = [mask_data["segmentation"] for mask_data in masks]

        # imInds1_ind, regInds1_ind, segMask1_ind = getIdxSingleFast(idx, masks_seg)
        # # imInds1 = np.concatenate((imInds1, imInds1_ind))
        # if self.order: 
        #     adjMat1_ind = nbrMasksAGGFastSingle(masks_seg, self.order).to(self.device)
        # else:
        #     adjMat1_ind = None

        # with torch.no_grad():
        #     # dino_desc = self.backbone(img.unsqueeze(0))
        #     dino_desc = self.backbone(x)
        # gd, _ = seg_vlad_gpu_single(self.ind_matrix, dino_desc, segMask1_ind, self.vlad.c_centers, self.desired_height, self.desired_width, desc_dim=1536, adj_mat=adjMat1_ind)#, device=self.device)
        gd, imInds1_ind, regInds1_ind, segMask1_ind = self.extract_vlad_features(x, idx).values()
        if self.pca is not None:
            gd = (
                self.pca.transform(gd.detach().cpu().numpy())
                if self.pca is not None
                else gd.cpu().numpy()
            )
            gd = torch.from_numpy(gd).float()
        
        return dict(
            out=gd,
            # imInds1=imInds1,
            imInds1_ind=imInds1_ind,
            regInds1_ind=regInds1_ind,
            segMask1_ind=segMask1_ind,
        )

    @torch.no_grad()
    def training_step(self, batch, batch_idx):
        if self._is_fitted:
            return
        if DEBUG > 1 and batch_idx > 5:
            return
        if batch_idx % self.fit_step != 0:
            return
        if len(self.features) > 50000:
            return
        # x = self.backbone(batch["image"])
        x, _, _, _ = self.extract_vlad_features(batch["image"], batch_idx).values()
        # x = einops.rearrange(x, "b c h w -> b (h w) c")
        self.features.append(x.detach())

    @torch.no_grad()
    def _fit(self):
        if self._is_fitted:
            return
        self.features = torch.vstack(self.features)
        # if self.vlad.c_centers is None:
        #     print("Fitting VLAD features...")
        #     self.features = self.vlad.fit_and_generate(self.features)
        # else:
        #     self.features = self.vlad.generate_multi(self.features)

        if self.pca is not None:
            print("Fitting PCA...")
            self.features = self.pca.fit_transform(self.features.cpu())
        del self.features


############################################################################
###   Helper functions from https://github.com/AnyLoc/Revisit-Anything   ###
############################################################################


def seg_vlad_gpu_single(
    ind, dino_desc, segMask, c_centers, desired_height, desired_width, desc_dim=1536, adj_mat=None#, device="cuda"
):
    device = dino_desc.device
    dh = desired_height // 14
    dw = desired_width // 14

    total_elements = dino_desc.shape[2] * dino_desc.shape[3]
    # dino_desc = dino_desc.reshape(1, desc_dim, total_elements).to("cuda")
    dino_desc = dino_desc.reshape(1, desc_dim, total_elements)

    dino_desc_norm = torch.nn.functional.normalize(dino_desc, dim=1)

    # IMPORTANT: could be wrong here
    # mask = torch.from_numpy(np.array(segMask)).to("cuda")
    mask = torch.from_numpy(np.array(segMask))
    mask = (
        torch.nn.functional.interpolate(
            mask.float().unsqueeze(0),
            [desired_height, desired_width],
            mode="nearest",
        )
        .squeeze()
        .bool()
        .reshape(len(segMask), -1)
    )
    mask_idx = torch.zeros((len(segMask), dh * dw), device=device).bool()
    mask_ind_flat = torch.argwhere(mask)
    mask_idx[mask_ind_flat[:, 0], ind[mask_ind_flat[:, 1]]] = True

    reg_feat_per = dino_desc_norm.permute(0, 2, 1)
    if adj_mat is not None:
        gd, execution_time = vlad_single(
            reg_feat_per.squeeze(),
            # c_centers.to(device),
            c_centers,
            mask_idx,
            # adj_mat.to(device),
            adj_mat,
        )
    else:
        gd, execution_time = vlad_single(
            # reg_feat_per.squeeze(), c_centers.to(device), mask_idx
            reg_feat_per.squeeze(), c_centers, mask_idx
        )

    return gd.cpu(), execution_time


def vlad_single(query_descs, c_centers, masks, adj_mat=None):#, device="cuda"):
    device = query_descs.device
    num_clusters = 32
    c_centers_norm = F.normalize(c_centers, dim=1).to(device)
    # c_centers_norm = F.normalize(c_centers, dim=1)
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
    num_c, masks, res, clus_labels, adjMat=None#, device="cuda"
):
    """
    Expects input tensors to be cuda and float/double
    """
    start_time = time.time()
    device = res.device
    vlads = []
    num_m = len(masks)

    if adjMat is None:
        adjMat = torch.eye(num_m, dtype=masks.dtype, device=masks.device)

    for li in range(num_c):
        inds_li = torch.where(clus_labels == li)[0].to(device)
        # inds_li = torch.where(clus_labels == li)[0]
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


def process_single_SAM(img, mask_generator, resize=True):
    if resize:
        w, h = img.shape[1], img.shape[0]
        width_SAM, height_SAM =  int(0.5 * w), int(0.5 * h)
        img = cv2.resize(img, (width_SAM, height_SAM))
    masks = mask_generator.generate(img)
    return masks


def loadSAM(sam_checkpoint, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


# def apply_pca_transform_from_pkl(data_tensor, pca_model_path):
#     """
#     Loads a PCA model from a pickle file and applies the transform to the given data tensor.

#     Parameters:
#     - data_tensor: A PyTorch tensor containing the data to be transformed.
#     - pca_model_path: Path to the pickle file containing the fitted PCA model.

#     Returns:
#     - A PyTorch tensor containing the transformed data.
#     """
#     # Ensure data is on CPU and converted to a NumPy array for PCA transformation
#     data_np = data_tensor.cpu().numpy()

#     # Load the fitted PCA model from disk
#     with open(pca_model_path, "rb") as file:
#         pca = pickle.load(file)

#     # Apply the PCA transform to the data
#     transformed_data_np = pca.transform(data_np)

#     # Convert the transformed data back to a PyTorch tensor
#     transformed_data_tensor = torch.from_numpy(transformed_data_np)

#     return transformed_data_tensor
