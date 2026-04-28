import torch
import cv2

import torchvision.transforms.functional as TF

# from geoloc.third_party.RoMa.romatch import roma_outdoor
from romatch import roma_outdoor
from loma.loma import to_pixel_coords, filter_matches, LoMa, LoMaB, LoMaR

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

        if kptA.shape[0] >= 4 and kptB.shape[0] >= 4:
            H, mask = self.ransac(kptA, kptB)
        else:
            H = None
            mask = None

        H = torch.from_numpy(H).float() if H is not None else torch.eye(3)
        if mask is not None:
            mask = torch.from_numpy(mask).bool()
        else:
            mask = torch.zeros((kptA.shape[0], 1), dtype=torch.bool)

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
        self.do_compile = do_compile

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

    def forward(self, qry_batch, ref_batch, **kwargs):
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


class LoMaMatcher(torch.nn.Module):
    def __init__(self, name="loma_R", resolution=448, do_compile=False, filter_threshold=0.1, num_sample_keypoints=2048):
        super().__init__()
        if name == "loma_R":
            cfg = LoMaR(compile=do_compile)
        else:
            cfg = LoMaB(compile=do_compile)
        self.matcher = LoMa(cfg)

        self.coarse_res = resolution # NOTE: strictly for compatibility with RomaMatcher, LoMa doesn't actually have a coarse stage
        self.do_compile = do_compile
        self.num_sample_keypoints = num_sample_keypoints
        self.filter_threshold = filter_threshold

    def forward(self, qry_batch, ref_batch, repeated_qry_optimization=False, **kwargs):
        qB, qC, qH, qW = qry_batch.shape
        rB, rC, rH, rW = ref_batch.shape        
        qry_batch = TF.resize(qry_batch, size=(self.coarse_res, self.coarse_res))
        ref_batch = TF.resize(ref_batch, size=(self.coarse_res, self.coarse_res))

        # Hacky way to only run on one query image
        if repeated_qry_optimization:
            keypoints_A, descriptors_A, h1, w1 = self.matcher.detect_and_describe(qry_batch[0:1], self.num_sample_keypoints)
            keypoints_A = keypoints_A.expand(qB, -1, -1)
            descriptors_A = descriptors_A.expand(qB, -1, -1)
        else:
            keypoints_A, descriptors_A, h1, w1 = self.matcher.detect_and_describe(qry_batch, self.num_sample_keypoints)

        with torch.no_grad():
            keypoints_B, descriptors_B, h2, w2 = self.matcher.detect_and_describe(ref_batch, self.num_sample_keypoints)

            scores = self.matcher(keypoints_A, keypoints_B, descriptors_A, descriptors_B)["scores"]
            m0, _, _, _ = filter_matches(scores, self.filter_threshold)

            all_kptsA = []
            all_kptsB = []
            # all_valid = []
            for i in range(qry_batch.shape[0]):
                valid = m0[i] > -1

                matched_A = keypoints_A[i][torch.where(valid)[0]]
                matched_B = keypoints_B[i][m0[i][valid]]

                # matched_A = to_pixel_coords(matched_A, h1, w1)
                # matched_B = to_pixel_coords(matched_B, h2, w2)
                matched_A = to_pixel_coords(matched_A, qH, qW)
                matched_B = to_pixel_coords(matched_B, rH, rW)

                matched_A = torch.nn.functional.pad(matched_A, (0, 0, 0, self.matcher.cfg.num_keypoints - matched_A.shape[0]), value=-1.0)
                matched_B = torch.nn.functional.pad(matched_B, (0, 0, 0, self.matcher.cfg.num_keypoints - matched_B.shape[0]), value=-1.0)

                all_kptsA.append(matched_A)
                all_kptsB.append(matched_B)
                # all_valid.append(valid)

            matched_A = torch.stack(all_kptsA, dim=0)
            matched_B = torch.stack(all_kptsB, dim=0)
            # valid = torch.stack(all_valid, dim=0)

            return matched_A, matched_B


def estimate_homography(kptsA, kptsB, ransac):
    Hs = []
    masks = []
    for kA, kB in zip(kptsA, kptsB):
        # Remove invalid keypoints
        valid_mask = (kA[:, 0] >= 0) & (kA[:, 1] >= 0) & (kB[:, 0] >= 0) & (kB[:, 1] >= 0)
        kA = kA[valid_mask]
        kB = kB[valid_mask]
        H, mask = ransac(kA.float(), kB.float())
        Hs.append(H.squeeze(0))
        # add padding to mask to make it the same length as the original keypoints
        mask = torch.nn.functional.pad(mask, (0, 0, 0, kptsA.shape[1] - mask.shape[0]), value=False)
        masks.append(mask.squeeze(0))
    num_inliers = torch.stack([m.sum() for m in masks])
    if len(Hs) > 0:
        Hs = torch.stack(Hs, dim=0)
    if len(masks) > 0:
        masks = torch.stack(masks, dim=0)
    return Hs, masks, num_inliers


def rerank(
    matcher, ransac, qry_image, ref_image_dataset, inds, dists=None, device=None, batch_size=1
):
    assert matcher is not None, "Matcher model must be provided"

    if dists is None:
        dists = torch.ones_like(inds).float() * -1.0

    if device is None:
        device = matcher.device

    if qry_image.ndim == 3:
        qry_image = qry_image.unsqueeze(0)

    mapped_inds = [int(idx % len(ref_image_dataset)) for idx in inds[0]]
    subset = torch.utils.data.Subset(ref_image_dataset, mapped_inds)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_with_geometry)

    all_num_inliers = []
    all_homographies = []
    all_num_outliers = []
    all_kptsA = []
    all_kptsB = []
    all_masks = []
    for batch in dataloader:
        ref_batch = batch["image"].to(device)
        qry_batch = qry_image.expand(ref_batch.shape[0], -1, -1, -1)

        kptsA, kptsB = matcher(qry_batch, ref_batch, repeated_qry_optimization=True)
        Hs, masks, num_inliers = estimate_homography(kptsA, kptsB, ransac)
        num_outliers = kptsA.shape[1] - num_inliers

        all_num_inliers.append(num_inliers)
        all_homographies.append(Hs)
        all_num_outliers.append(num_outliers)
        all_kptsA.append(kptsA)
        all_kptsB.append(kptsB)
        all_masks.append(masks)
        
    all_num_inliers = torch.cat(all_num_inliers, dim=0).cpu()
    all_homographies = torch.cat(all_homographies, dim=0).cpu()
    all_num_outliers = torch.cat(all_num_outliers, dim=0).cpu()
    all_kptsA = torch.cat(all_kptsA, dim=0).cpu()
    all_kptsB = torch.cat(all_kptsB, dim=0).cpu()
    all_masks = torch.cat(all_masks, dim=0).cpu()

    sorted_order = torch.argsort(all_num_inliers, descending=True).numpy()
    inds = inds[:, sorted_order]
    dists = dists[:, sorted_order]
    all_num_inliers = all_num_inliers[sorted_order]
    all_num_outliers = all_num_outliers[sorted_order]
    all_homographies = all_homographies[sorted_order]
    all_kptsA = all_kptsA[sorted_order]
    all_kptsB = all_kptsB[sorted_order]
    all_masks = all_masks[sorted_order]

    return dict(
        inds=inds,
        dists=dists,
        all_num_inliers=all_num_inliers.tolist(),
        all_homographies=all_homographies.tolist(),
        all_num_outliers=all_num_outliers.tolist(),
        qry_kpts=all_kptsA.tolist(),
        ref_kpts=all_kptsB.tolist(),
        inliers=all_masks.tolist(),
    )