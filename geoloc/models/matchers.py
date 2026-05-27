import torch
import torchvision.transforms.functional as TF

# from geoloc.third_party.RoMa.romatch import roma_outdoor
from romatch import roma_outdoor
from loma.loma import to_pixel_coords, filter_matches, LoMa, LoMaB, LoMaR


class RomaMatchAnythingMatcher(torch.nn.Module):
    def __init__(
        self,
        weights_path,
        coarse_res=560,
        upsample_res=864,
        use_custom_corr=False,
        upsample_preds=False,
        symmetric=False,
        num_sample_keypoints=5000,
        device=None,
        do_compile=False,
        sample_mode="threshold_balanced",
    ):
        super().__init__()
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
            sample_mode=sample_mode,
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
                device=self.device,
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

        self.coarse_res = resolution  # NOTE: strictly for compatibility with RomaMatcher, LoMa doesn't actually have a coarse stage
        self.do_compile = do_compile
        self.num_sample_keypoints = num_sample_keypoints
        self.filter_threshold = filter_threshold

    def forward(self, qry_batch, ref_batch, repeated_qry_optimization=False, **kwargs):
        qB, qC, qH, qW = qry_batch.shape
        rB, rC, rH, rW = ref_batch.shape
        qry_batch = TF.resize(qry_batch, size=(self.coarse_res, self.coarse_res))
        ref_batch = TF.resize(ref_batch, size=(self.coarse_res, self.coarse_res))

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
            for i in range(qry_batch.shape[0]):
                valid = m0[i] > -1

                matched_A = keypoints_A[i][torch.where(valid)[0]]
                matched_B = keypoints_B[i][m0[i][valid]]

                matched_A = to_pixel_coords(matched_A, qH, qW)
                matched_B = to_pixel_coords(matched_B, rH, rW)

                matched_A = torch.nn.functional.pad(matched_A, (0, 0, 0, self.matcher.cfg.num_keypoints - matched_A.shape[0]), value=-1.0)
                matched_B = torch.nn.functional.pad(matched_B, (0, 0, 0, self.matcher.cfg.num_keypoints - matched_B.shape[0]), value=-1.0)

                all_kptsA.append(matched_A)
                all_kptsB.append(matched_B)

            matched_A = torch.stack(all_kptsA, dim=0)
            matched_B = torch.stack(all_kptsB, dim=0)

            return matched_A, matched_B
