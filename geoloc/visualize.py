import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


DPI = 60


def visualize_top_k_retrieved(
    query_image,
    query_ix,
    inds,
    ref_image_dataset,
    savedir,
    filename=None,
    gdists=None,
    min_dists_at_k=None,
    gt_pos=None,
    rotexp_thetas=[0],
    query_theta=0,
    rotator_theta=None,
    rotator_ref_thetas=None,
    all_num_inliers=None,
    all_num_outliers=None,
    passed_distribution_check=None,
    idcscore=None,
):
    fig, axes = plt.subplots(1, 6, figsize=(15, 4))

    if rotator_theta is not None:
        rotator_theta  = rotator_theta * 180.0 / torch.pi
        query_image = TF.rotate(query_image, rotator_theta)
    else:
        rotator_theta = 0
    query_image = query_image.cpu().numpy() * 255
    query_image = query_image.astype(np.uint8)

    axes[0].imshow(query_image.transpose(1, 2, 0))
    # axes[0].set_title(f"Qry | rot: {query_theta:.2f}° | rrot: {rotator_theta:.2f}°") # Uncomment this if every needed
    axes[0].set_title(f"Qry idx: {query_ix}", fontsize=10)
    axes[0].axis("off")

    for j in range(1, 6):
        ref_im_ix = inds[0, j - 1] % len(ref_image_dataset)
        ref_im = ref_image_dataset[ref_im_ix.item()]["image"]

        rot_ix = inds[0, j - 1] // len(ref_image_dataset)
        rot = rotexp_thetas[rot_ix]
        ref_im = TF.rotate(ref_im, rot)

        if rotator_ref_thetas is not None:
            ref_rotator_theta = rotator_ref_thetas[ref_im_ix]
            ref_rotator_theta = ref_rotator_theta * 180.0 / np.pi
            ref_im = TF.rotate(ref_im, ref_rotator_theta)
        else:
            ref_rotator_theta = 0

        ref_im = TF.resize(ref_im, (query_image.shape[1], query_image.shape[2]))
        ref_im = ref_im.cpu().numpy() * 255
        ref_im = ref_im.astype(np.uint8)

        title_parts = []
        if gdists is not None:
            title_parts.append(f"{gdists[j-1]:.2f} m")

        ratio_text = None
        if all_num_inliers is not None and all_num_outliers is not None:
            inliers = int(all_num_inliers[j - 1])
            outliers = int(all_num_outliers[j - 1])
            ratio = inliers / max(outliers, 1)
            ratio_text = f"I/O: {inliers}/{outliers} ({ratio:.2f})"

        ref_title = f"Ref idx: {ref_im_ix} | " + " | ".join(title_parts)

        axes[j].imshow(ref_im.transpose(1, 2, 0))
        axes[j].set_title(ref_title, fontsize=10)
        axes[j].set_xticks([])
        axes[j].set_yticks([])
        if ratio_text is not None:
            axes[j].text(
                0.5,
                -0.08,
                ratio_text,
                transform=axes[j].transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )

        # Add colored border for TP/FP if gt_pos is provided
        if gt_pos is not None:
            color = "#00FF00" if ref_im_ix in gt_pos else "#FF0000"  # vibrant green/red
            for spine in axes[j].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
    
    if passed_distribution_check is not None:
        color = "green" if passed_distribution_check else "red"
        text = f"PASSED ({idcscore:.2f})" if passed_distribution_check else f"FAILED ({idcscore:.2f})"
        fig.text(0.05, 0.90, text, fontsize=10, color=color, ha="left", va="bottom")

    title = f"Query #{query_ix} theta: {query_theta}"
    if min_dists_at_k is not None:
        title = (
            f"Query #{query_ix} theta: {query_theta} | "
            f"min dist@1: {min_dists_at_k[0]:.2f} m | "
            f"min dist@5: {min_dists_at_k[1]:.2f} m"
        )

    fig.suptitle(title, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if filename is None:
        filename = f"query_{query_ix:03d}_rot_{query_theta:03d}.png"
    save_path = os.path.join(savedir, filename)
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)


def _color_map_color(value, cmap_name="jet", vmin=0, vmax=1):
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[
        :3
    ]  # will return rgba, we take only first 3 so we get rgb
    return rgb


def visualize_vlad_clusters(
    query_image, query_ix, query_residuals, vlad, savedir, query_theta=0
):
    fig, ax = plt.subplots()
    colors = np.zeros((vlad.num_clusters, 3))
    for j in range(vlad.num_clusters):
        colors[j, :] = _color_map_color(j / (vlad.num_clusters - 1))

    # Loop through all the patches inside image (Dino patches)
    img_desc = []
    for j in range(query_residuals.shape[1]):
        cur_res_vec = torch.abs(query_residuals[0, j])
        res_idx = torch.argmin(torch.sum(cur_res_vec, dim=1))
        img_desc.append(res_idx)
    img_desc = np.asarray(img_desc)

    # Color based on the closest clusters
    desc_color = np.zeros((img_desc.shape[0], 3))
    for c in range(vlad.num_clusters):
        img_idx = np.argwhere(img_desc == c)
        desc_color[img_idx[:, 0]] = colors[c] * 255

    # Merge cluster color map with original image
    lh = lw = int(np.sqrt(img_desc.shape[0]))
    vlad_color_img = np.reshape(desc_color, (lh, lw, 3))
    vlad_color_img = cv2.resize(
        vlad_color_img,
        (query_image.shape[1], query_image.shape[2]),
        interpolation=cv2.INTER_NEAREST,
    )

    vlad_color_img = vlad_color_img.astype(np.uint8)
    query_image = query_image.cpu().numpy() * 255
    query_image = query_image.astype(np.uint8)

    ax.imshow(vlad_color_img, alpha=0.7)
    ax.imshow(query_image.transpose(1, 2, 0), alpha=0.5)
    ax.set_title(f"VLAD Color Map")
    ax.axis("off")

    save_path = os.path.join(
        savedir, f"query_{query_ix:03d}_rot_{query_theta:03d}_vladvis.png"
    )
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)


def visualize_similarity_heatmap_gurs(
    dists,
    inds,
    ref_image_dataset,
    query_ix,
    query_theta,
    query_northing,
    query_easting,
    savedir,
    rotref_exp=False,
):
    fig, ax = plt.subplots()

    # The tiles in the reference dataset are order by column top-to-bottom left-to-right
    heatmap_w = ref_image_dataset.num_tiles_width
    heatmap_h = ref_image_dataset.num_tiles_height
    heatmap = np.zeros((heatmap_h, heatmap_w))
    minx, miny, maxx, maxy = ref_image_dataset.disc_border_polygon.bounds

    # Calculate the ground truth position in the heatmap
    gt_row = int((maxy - query_northing) / (maxy - miny) * heatmap_h)
    gt_col = int((query_easting - minx) / (maxx - minx) * heatmap_w)
    ax.plot(gt_col, gt_row, marker="+", color="green", markersize=10, markeredgewidth=1)

    for h, (dist, ind) in enumerate(zip(dists[0], inds[0])):
        if rotref_exp:
            ind = ind % len(ref_image_dataset)
        # Calculate the retrieved image position in the heatmap
        pr_east, pr_north = ref_image_dataset.get_coords_only(ind)
        pr_row = int((maxy - pr_north) / (maxy - miny) * heatmap_h)
        pr_col = int((pr_east - minx) / (maxx - minx) * heatmap_w)
        heatmap[pr_row, pr_col] = dist
        if h == 0:
            ax.plot(
                pr_col,
                pr_row,
                marker="x",
                color="blue",
                markersize=10,
                markeredgewidth=1,
            )

    # Normalize heatmap
    heatmap = np.where(heatmap == 0, np.nan, heatmap)
    minval = np.nanmin(heatmap)
    maxval = np.nanmax(heatmap)
    heatmap = (heatmap - minval) / (maxval - minval)
    np.save(
        os.path.join(savedir, f"heatmap_{query_ix:03d}_rot_{query_theta:03d}.npy"),
        heatmap,
    )

    # Plot heatmap
    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad(color=(0, 0, 0, 0))  # RGBA: fully transparent
    ax.imshow(heatmap, cmap=cmap, interpolation="nearest")
    ax.set_title(f"Heatmap")
    ax.axis("off")

    save_path = os.path.join(
        savedir, f"query_{query_ix:03d}_rot_{query_theta:03d}_heatmap.png"
    )
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)


def visualize_similarity_heatmap_visloc(
    dists,
    inds,
    ref_image_dataset,
    query_ix,
    query_theta,
    savedir,
):
    fig, ax = plt.subplots()

    # The tiles in the reference dataset are order by column top-to-bottom left-to-right
    heatmap_w = ref_image_dataset.num_tiles_width
    heatmap_h = ref_image_dataset.num_tiles_height
    heatmap = np.zeros((heatmap_h, heatmap_w))

    for i, (dist, ind) in enumerate(zip(dists[0], inds[0])):
        # convert ind to position in heatmap
        row = ind % heatmap_h
        col = ind // heatmap_h
        heatmap[row, col] = dist
        if i == 0:
            ax.plot(
                col, row, marker="x", color="cyan", markersize=10, markeredgewidth=2
            )

    # Normalize heatmap
    heatmap = np.where(heatmap == 0, np.nan, heatmap)
    minval = np.nanmin(heatmap)
    maxval = np.nanmax(heatmap)
    heatmap = (heatmap - minval) / (maxval - minval)
    np.save(
        os.path.join(savedir, f"heatmap_{query_ix:03d}_rot_{query_theta:03d}.npy"),
        heatmap,
    )

    # Plot heatmap
    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad(color=(0, 0, 0, 0))  # RGBA: fully transparent
    ax.imshow(heatmap, cmap=cmap, interpolation="nearest")
    ax.set_title(f"Heatmap")
    ax.axis("off")

    save_path = os.path.join(
        savedir, f"query_{query_ix:03d}_rot_{query_theta:03d}_heatmap.png"
    )
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)


def visualize_matches(im_A, im_B, kpts_A, kpts_B, save_path=None, multicolor=False, line_alpha=0.5):
    h1, w1 = im_A.shape[-2:]
    h2, w2 = im_B.shape[-2:]

    im_A = im_A.detach().cpu().numpy() * 255
    im_B = im_B.detach().cpu().numpy() * 255
    im_A = im_A.astype(np.uint8)
    im_B = im_B.astype(np.uint8)

    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)))
    canvas.paste(Image.fromarray(im_A.transpose(1, 2, 0)), (0, 0))
    canvas.paste(Image.fromarray(im_B.transpose(1, 2, 0)), (w1, 0))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rng = np.random.default_rng(0)

    line_alpha = int(line_alpha * 255)
    color = (0, 255, 0, line_alpha)  # default green with alpha
    for (x1, y1), (x2, y2) in zip(kpts_A, kpts_B):
        if multicolor:
            r, g, b = rng.integers(0, 256, 3).tolist()
            color = (r, g, b, line_alpha)
        draw.line([(x1, y1), (x2 + w1, y2)], fill=color, width=1)

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(save_path)
        print(f"Saved {len(kpts_A)} matches to {save_path}")
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas)
        plt.axis("off")
        plt.show()


def visualize_warp(im_A, im_B, H, save_path=None):
    """
    Warps im_A onto im_B using homography H and shows the overlap as a blend.
    im_A, im_B: torch tensors of shape (C, H, W) in [0, 1]
    H: 3x3 homography matrix (numpy or torch)
    """
    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy()

    h1, w1 = im_A.shape[-2:]
    h2, w2 = im_B.shape[-2:]

    im_A_np = (im_A.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    im_B_np = (im_B.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)

    # Warp im_A into the coordinate frame of im_B
    warped_A = cv2.warpPerspective(im_A_np, H, (w2, h2))

    # Build overlap mask: pixels where warped_A is non-zero
    mask = (warped_A.sum(axis=-1) > 0).astype(np.uint8)

    # Blend warped_A and im_B in the overlapping region
    blended = im_B_np.copy()
    blended[mask == 1] = (
        0.5 * warped_A[mask == 1] + 0.5 * im_B_np[mask == 1]
    ).astype(np.uint8)

    # Highlight the overlap boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 0), 2)

    result = Image.fromarray(blended)

    plt.figure(figsize=(10, 10))
    plt.imshow(result)
    plt.axis("off")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def visualize_matches2(imgA_tensor, imgB_tensor, kptsA, kptsB, mask=None, max_matches=200, skip_draw=False, always_draw=False, save_path=None):
    """
    Visualize matches between two images side by side.
    
    Args:
        imgA_tensor, imgB_tensor: Tensors representing the images (e.g., from a model)
        kptsA, kptsB: Keypoints in pixel coordinates (torch tensors or numpy arrays)
        kptsA, kptsB: Keypoints in pixel coordinates (torch tensors or numpy arrays)
        mask: RANSAC inlier mask (optional) — if provided, colors inliers green, outliers red
        max_matches: Max number of matches to draw (for readability)
    """
    def tensor_to_numpy(t):
        t = t.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)          # (1, 3, H, W) -> (3, H, W)
        t = t.permute(1, 2, 0)        # (3, H, W) -> (H, W, 3)
        t = t.numpy()
        if t.max() <= 1.1:
            t = (t * 255)
        return t.astype(np.uint8)

    imgA = tensor_to_numpy(imgA_tensor)
    imgB = tensor_to_numpy(imgB_tensor)

    # Convert tensors to numpy if needed
    ptsA = kptsA.cpu().numpy() if hasattr(kptsA, 'cpu') else kptsA
    ptsB = kptsB.cpu().numpy() if hasattr(kptsB, 'cpu') else kptsB

    # Resize images to same height for side-by-side display
    H = max(imgA.shape[0], imgB.shape[0])
    def pad_to_height(img, H):
        h, w = img.shape[:2]
        padded = np.zeros((H, w, 3), dtype=np.uint8)
        padded[:h, :w] = img
        return padded

    imgA = pad_to_height(imgA, H)
    imgB = pad_to_height(imgB, H)

    W_A = imgA.shape[1]
    canvas = np.concatenate([imgA, imgB], axis=1)

    # Determine inlier/outlier split
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        inlier_mask = mask.ravel().astype(bool)
        inlier_pts_A = ptsA[inlier_mask]
        inlier_pts_B = ptsB[inlier_mask]
        outlier_pts_A = ptsA[~inlier_mask]
        outlier_pts_B = ptsB[~inlier_mask]
    else:
        inlier_pts_A, inlier_pts_B = ptsA, ptsB
        outlier_pts_A = outlier_pts_B = np.empty((0, 2))

    pct = len(inlier_pts_A) / len(ptsA) * 100 if len(ptsA) > 0 else 0
    if (not skip_draw and pct > 0.20 and len(inlier_pts_A) > 50) or always_draw:  # Only draw if we have a reasonable number of inliers
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.imshow(canvas)
        ax.set_title(f"Inliers: {len(inlier_pts_A)}  |  Outliers: {len(outlier_pts_A)}", fontsize=13)
        ax.axis("off")

        def draw_matches(pts_A, pts_B, color, alpha=0.6, max_n=max_matches):
            if len(pts_A) == 0:
                return
            # Subsample for readability
            idx = np.random.choice(len(pts_A), min(max_n, len(pts_A)), replace=False)
            for i in idx:
                xA, yA = pts_A[i]
                xB, yB = pts_B[i][0] + W_A, pts_B[i][1]
                ax.plot([xA, xB], [yA, yB], color=color, alpha=alpha, linewidth=0.6)
                ax.scatter([xA], [yA], c=[color], s=4, zorder=5)
                ax.scatter([xB], [yB], c=[color], s=4, zorder=5)

        # Draw outliers first (behind), then inliers on top
        draw_matches(outlier_pts_A, outlier_pts_B, color="red",   alpha=0.3, max_n=100)
        draw_matches(inlier_pts_A,  inlier_pts_B,  color="lime",  alpha=0.7, max_n=max_matches)

        plt.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
