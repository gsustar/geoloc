import torch
import cv2

import torchvision.transforms.functional as TF

from geoloc.third_party.RoMa.romatch import roma_outdoor
from geoloc.data.utils import collate_with_geometry


class OpenCVRANSAC:
    def __init__(self, reproj_threshold=1.0, maxIters=10000):
        self.ransac = (lambda kptA, kptB: 
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
        H, mask = self.ransac(kptA, kptB)
        if H is not None:
            H = torch.from_numpy(H).float()
        else:
            H = torch.eye(3)
        if mask is not None:
            mask = torch.from_numpy(mask).bool()
        else:
            mask = torch.zeros(kptA.shape[0], dtype=torch.bool)
        return H, mask


class RomaMatchAnythingMatcher(torch.nn.Module):
    def __init__(
        self, 
        # ransac,
        weights_path,
        coarse_res=560, 
        upsample_res=864, 
        use_custom_corr=False, 
        upsample_preds=False, 
        symmetric=False,
        num_sample_keypoints=5000,
        device=None,
        do_compile=False,
        sample_mode="threshold_balanced"
    ):
        super().__init__()
        # self.ransac = ransac
        self.weights_path = weights_path
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_size = 14
        self.coarse_res = coarse_res
        self.upsample_res = upsample_res
        self.upsample_preds = upsample_preds
        self.num_sample_keypoints = num_sample_keypoints

        self.matcher = roma_outdoor(
            device=self.device, 
            coarse_res=coarse_res, 
            upsample_res=upsample_res, 
            use_custom_corr=use_custom_corr, 
            upsample_preds=upsample_preds, 
            symmetric=symmetric,
            amp_dtype=torch.float16,
            do_compile=do_compile,
            sample_mode=sample_mode
        )
        ma_roma_state_dict = torch.load(self.weights_path)["state_dict"]
        new_roma_state_dict = {}
        for key, value in ma_roma_state_dict.items():
            new_key = key.replace("matcher.model.", "")
            new_roma_state_dict[new_key] = value
        self.matcher.load_state_dict(new_roma_state_dict, strict=False)

    def forward(self, qry_batch, ref_batch):
        qB, qC, qH, qW = qry_batch.shape
        rB, rC, rH, rW = ref_batch.shape
        all_kptsA = []
        all_kptsB = []

        high_res_qry_batch = None
        high_res_ref_batch = None
        if self.upsample_preds:
            assert qH == self.upsample_res and qW == self.upsample_res, "Query batch must be at upsample resolution when upsample_preds is True"
            assert rH == self.upsample_res and rW == self.upsample_res, "Reference batch must be at upsample resolution when upsample_preds is True"
            high_res_qry_batch = qry_batch
            high_res_ref_batch = ref_batch
        coarse_qry_batch = TF.resize(qry_batch, size=self.coarse_res)
        coarse_ref_batch = TF.resize(ref_batch, size=self.coarse_res)

        with torch.no_grad():
            warps, certainties = self.matcher.match(
                im_A_input=coarse_qry_batch, 
                im_B_input=coarse_ref_batch,
                im_A_high_res=high_res_qry_batch,
                im_B_high_res=high_res_ref_batch,
                batched=True, 
                device=self.device
            )
            for i in range(rB):
                matches, certainty = self.matcher.sample(warps[i], certainties[i], self.num_sample_keypoints)
                kptsA, kptsB = self.matcher.to_pixel_coordinates(matches, qH, qW, rH, rW)
                all_kptsA.append(kptsA)
                all_kptsB.append(kptsB)

        all_kptsA = torch.stack(all_kptsA, dim=0)
        all_kptsB = torch.stack(all_kptsB, dim=0)
        return all_kptsA, all_kptsB


def estimate_homography(kptsA, kptsB, ransac):
    Hs = []
    masks = []
    for kA, kB in zip(kptsA, kptsB):
        H, mask = ransac(kA.float(), kB.float())
        Hs.append(H.squeeze(0))
        masks.append(mask.squeeze(0))
    num_inliers = torch.stack([m.sum() for m in masks])
    if len(Hs) > 0:
        Hs = torch.stack(Hs, dim=0)
    if len(masks) > 0:
        masks = torch.stack(masks, dim=0)
    return Hs, masks, num_inliers


def rerank(
    matcher, ransac, qry_image, ref_image_dataset, inds, dists=None, device=None, batch_size=1, rotations=None
):
    assert matcher is not None, "Matcher model must be provided"

    if dists is None:
        dists = torch.ones_like(inds).float() * -1.0

    if device is None:
        device = matcher.device

    if qry_image.ndim == 3:
        qry_image = qry_image.unsqueeze(0)

    if rotations is None or len(rotations) == 0:
        rotations = [0]

    mapped_inds = [int(idx % len(ref_image_dataset)) for idx in inds[0]]
    subset = torch.utils.data.Subset(ref_image_dataset, mapped_inds)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_with_geometry)

    all_num_inliers = []
    all_homographies = []
    all_num_outliers = []
    for batch in dataloader:
        ref_batch = batch["image"].to(device)
        qry_batch_base = qry_image.expand(ref_batch.shape[0], -1, -1, -1)

        best_num_inliers = None
        best_homographies = None
        num_matches = None

        for angle in rotations:
            qry_batch = TF.rotate(qry_batch_base, angle) if angle != 0 else qry_batch_base
            kptsA, kptsB = matcher(qry_batch, ref_batch)
            Hs, masks, num_inliers = estimate_homography(kptsA, kptsB, ransac)

            if masks.ndim >= 2:
                num_matches = masks.shape[1]

            if best_num_inliers is None:
                best_num_inliers = num_inliers
                best_homographies = Hs
            else:
                better = num_inliers > best_num_inliers
                best_num_inliers = torch.where(better, num_inliers, best_num_inliers)
                better_h = better.view(-1, 1, 1)
                best_homographies = torch.where(better_h, Hs, best_homographies)

        if num_matches is None:
            num_matches = 0

        best_num_outliers = num_matches - best_num_inliers
        all_num_inliers.append(best_num_inliers)
        all_homographies.append(best_homographies)
        all_num_outliers.append(best_num_outliers)
    all_num_inliers = torch.cat(all_num_inliers, dim=0).cpu()
    all_homographies = torch.cat(all_homographies, dim=0).cpu()
    all_num_outliers = torch.cat(all_num_outliers, dim=0).cpu()

    sorted_order = torch.argsort(all_num_inliers, descending=True).numpy()
    inds = inds[:, sorted_order]
    dists = dists[:, sorted_order]
    all_num_inliers = all_num_inliers[sorted_order]
    all_num_outliers = all_num_outliers[sorted_order]
    all_homographies = all_homographies[sorted_order]

    return dict(
        inds=inds,
        dists=dists,
        all_num_inliers=all_num_inliers.tolist(),
        all_homographies=all_homographies.tolist(),
        all_num_outliers=all_num_outliers.tolist(),
    )