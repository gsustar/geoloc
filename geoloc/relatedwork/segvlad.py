import torch
import numpy as np
import torch.nn.functional as F
import time


class SegVLAD:

    def __init__(self):
        pass

    def forward(self, image):

        # Get all SAM masks for the image

        # Get tuple of (img_idx, mask_idx)

        # Combine masks into super-segments

        # Get DINOv2 descriptors for the image

        # Compute Hard-VLAD residuals for each

        pass


def seg_vlad_gpu_single(
    ind,
    idx,
    desc_path_in,
    img_key,
    segMask,
    c_centers,
    cfg,
    desc_dim=1536,
    adj_mat=None,
):
    vlad_dim = 32 * desc_dim
    dh = cfg["desired_height"] // 14
    dw = cfg["desired_width"] // 14

    dino_desc = torch.from_numpy(desc_path_in[img_key]["ift_dino"][()])
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
            idx,
            mask_idx,
            adj_mat.to("cuda"),
        )
    else:
        gd, execution_time = vlad_single(
            reg_feat_per.squeeze(), c_centers.to("cuda"), idx, mask_idx
        )

    return gd.cpu()


def vlad_single(query_descs, c_centers, idx, masks, adj_mat=None):
    num_clusters = 32

    c_centers_norm = F.normalize(c_centers, dim=1).to("cuda")
    labels = torch.argmax(query_descs @ c_centers_norm.T, dim=1)

    # TODO: use normed clusters
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


def getNbrsDelaunay(tri, v):
    indptr, indices = tri.vertex_neighbor_vertices
    v_nbrs = indices[indptr[v] : indptr[v + 1]]
    v_nbrs = [[v, u] for u in v_nbrs]
    return v_nbrs


from scipy.spatial import Delaunay


def nbrMasksAGGFastSingle(masks_seg, order=1):
    # img_key is img_name
    # masks = [mask_in[img_key + f'/masks/{k}/'] for k in natsorted(mask_in[img_key + '/masks/'].keys())]
    # mask_cords = np.array([np.array(np.nonzero(mask['segmentation'][()])).mean(1)[::-1] for mask in masks])
    # masks = [mask_in[img_key + f'/masks/{k}/'] for k in natsorted(mask_in[img_key + '/masks/'].keys())]
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

        # if order ==1:
        #     adj_mat.append(adj_mat.bool())
        # elif order==2:
        #     adj_mat.append((adj_mat@adj_mat).bool())
        # elif order==3:
        #     adj_mat.append((adj_mat@adj_mat@adj_mat).bool())
        # elif order==4:
        #     adj_mat.append((adj_mat@adj_mat@adj_mat@adj_mat).bool())
        # elif order==5:
        #     adj_mat.append((adj_mat@adj_mat@adj_mat@adj_mat@adj_mat).bool())

        adj_mat_power = adj_mat.clone()
        for _ in range(order - 1):  # Multiply original matrix order-1 times
            adj_mat_power = adj_mat_power @ adj_mat
        adj_mat = adj_mat_power.bool()

    else:
        # nbr_list = [0,1] #Important-2: In full function this is the code. Just trying out better version in single code.
        nbr_list = (
            [0, 1] if len(mask_cords) > 1 else [0]
        )  # Adjusting for cases with less than 2 masks
        for v in range(len(mask_cords)):
            adj_mat[v][nbr_list] = 1
        adj_mat = adj_mat.bool()

    return adj_mat


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


def getIdxSingleFast(img_idx, masks_seg, minArea=400, returnMask=True):
    imInds = []
    regIndsIm = []
    segmask = []
    count = 0

    # im_name = ims[img_idx]
    # key = f"{im_name}"

    # Preload all masks for the image
    # masks_seg, mask_keys = preload_masks(masks_in, key)

    for mask in masks_seg:
        # Assuming 'area' is a property that can be efficiently accessed without loading the entire mask
        # This may require adjusting based on your actual HDF5 structure and how 'area' is stored
        # area = masks_in[f"{key}/masks/{k}/area"][()]

        # if area > minArea:
        if returnMask:
            segmask.append(mask)
        regIndsIm.append(count)
        imInds.append(img_idx)
        count += 1

    return np.array(imInds), regIndsIm, segmask


def preload_masks(masks_in, image_key):
    """
    Preloads all masks for a given image key from the HDF5 file.

    Args:
    - masks_in: An open h5py File or Group object representing the HDF5 file or a group within it.
    - image_key: The key in the HDF5 file for the specific image.

    Returns:
    - A list of mask data loaded into memory.
    """
    masks_path = f"{image_key}/masks/"
    mask_keys = natsorted(masks_in[masks_path].keys())
    masks_seg = [masks_in[masks_path + k]["segmentation"][()] for k in mask_keys]
    return masks_seg


from tqdm import tqdm

print("Computing SegLoc for all images in the dataset...")
for r_id, r_img in tqdm(
    enumerate(ims1_r), total=len(ims1_r), desc="Processing for reference images..."
):

    # print(r_id, r_img)
    # Preload all masks for the image
    masks_seg = preload_masks(masks1_h5_r, r_img)

    imInds1_ind, regInds1_ind, segMask1_ind = getIdxSingleFast(
        r_id, masks_seg, minArea=experiment_config["minArea"]
    )

    imInds1 = np.concatenate((imInds1, imInds1_ind))

    if order:
        adjMat1_ind = nbrMasksAGGFastSingle(masks_seg, order)
    else:
        adjMat1_ind = None

    gd = seg_vlad_gpu_single(
        ind_matrix,
        idx_matrix,
        dino1_h5_r,
        r_img,
        segMask1_ind,
        c_centers,
        cfg,
        desc_dim=1536,
        adj_mat=adjMat1_ind,
    )

    if experiment_config["pca"]:
        batch_descriptors_r.append(gd)
        if (r_id + 1) % batch_size == 0 or (r_id + 1) == len(
            ims1_r
        ):  ## Once we have accumulated descriptors for 100 images or are at the last image, process the batch
            segFtVLAD1_batch = torch.cat(batch_descriptors_r, dim=0)
            # Reset batch descriptors for the next batch
            batch_descriptors_r = []

            print("Applying PCA to batch descriptors... at image ", r_id)
            segFtVLAD1Pca_batch = apply_pca_transform_from_pkl(
                segFtVLAD1_batch, pca_model_path
            )
            del segFtVLAD1_batch

            segFtVLAD1Pca_list.append(segFtVLAD1Pca_batch)

    else:
        segFtVLAD1_list.append(
            gd
        )  # imfts_batch same as gd here, in the full image function, it is for 100 images at a time
