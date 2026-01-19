import os
import torch
import einops
import numpy as np
import torchvision.transforms.functional as TF

from torchvision.transforms import InterpolationMode

from ..vprmodel import VPRModel
from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.model import RetrievalModel


class Mast3rRetrievalModel(VPRModel):
    """Mast3r retrieval model."""

    def __init__(self, **kwargs):
        super().__init__(backbone=None, aggregator=None, **kwargs)


    def load_from_legacy_checkpoint(self, checkpoint_path, **kwargs):
        assert os.path.isdir(checkpoint_path), "`checkpoint_path` must be a directory containing mast3r_retrieval and mast3r_backbone checkpoints"
        retrieval_checkpoint = os.path.join(checkpoint_path, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")
        backbone_checkpoint = os.path.join(checkpoint_path, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")

        print(f'Loading retrieval model from {retrieval_checkpoint}')
        ckpt = torch.load(retrieval_checkpoint, 'cpu', weights_only=False)
        ckpt_args = ckpt['args']

        backbone = AsymmetricMASt3R.from_pretrained(backbone_checkpoint)
        self.model = RetrievalModel(
            backbone, freeze_backbone=ckpt_args.freeze_backbone, prewhiten=ckpt_args.prewhiten,
            hdims=list(map(int, ckpt_args.hdims.split('_'))) if len(ckpt_args.hdims) > 0 else "",
            residual=getattr(ckpt_args, 'residual', False), postwhiten=ckpt_args.postwhiten,
            featweights=ckpt_args.featweights, nfeat=ckpt_args.nfeat
        ).eval()

        msg = self.model.load_state_dict(ckpt['model'], strict=False)
        assert all(k.startswith('backbone') for k in msg.missing_keys)
        assert len(msg.unexpected_keys) == 0
        self.imsize = ckpt_args.imsize
        return self

    def _resize(self, img, long_edge_size):
        H, W = img.shape[-2:]
        S = max(H, W)
        interpolation = InterpolationMode.BILINEAR if S > long_edge_size else InterpolationMode.BICUBIC
        scale = long_edge_size / S
        new_H = int(round(H * scale))
        new_W = int(round(W * scale))
        resized = TF.resize(
            img, 
            size=(new_H, new_W), 
            interpolation=interpolation,
            antialias=True
        )
        return resized

    def prepare_input(self, x, size, patch_size=16, square_ok=False):
        assert self.imsize == 512
        H1, W1 = x.shape[-2:]
        if size == 224:
            x = self._resize(x, round(size * max(W1/H1, H1/W1)))
        else:
            x = self._resize(x, size)

        H, W = x.shape[-2:]
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            crop_h = crop_w = 2 * half
        else:
            halfw = ((2 * cx) // patch_size) * patch_size / 2
            halfh = ((2 * cy) // patch_size) * patch_size / 2
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            crop_h = int(2 * halfh)
            crop_w = int(2 * halfw)

        top = int(cy - crop_h / 2)
        left = int(cx - crop_w / 2)
        x = TF.crop(x, top, left, crop_h, crop_w)

        W2, H2 = x.shape[-2:]
        x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return dict(
            img=x,
            true_shape=np.int32([H2, W2])
        )

    def forward(self, x, idx, max_nfeat_per_image=None, max_nfeat_per_image2=None, tocpu=False):
        bs = x.shape[0]
        assert bs == 1, "Batch size greater than 1 not supported in Mast3rRetrievalModel"
        x = self.prepare_input(x, size=self.imsize)
        feat, _, _ = self.model.forward_local(x)
        imids = torch.cat([torch.ones_like(feat[i, :, 0]).to(dtype=torch.int64) * (idx * bs + i) for i in range(bs)])
        feat = einops.rearrange(feat, "b n d -> (b n) d")

        # if max_nfeat_per_image is not None and feat.size(0) > max_nfeat_per_image: # NOTE: from original code, but not used
        #     feat = feat[torch.randperm(feat.size(0))[:max_nfeat_per_image], :]
        # if max_nfeat_per_image2 is not None and feat.size(0) > max_nfeat_per_image2:
        #     feat = feat[:max_nfeat_per_image2, :]
        if tocpu:
            feat = feat.cpu()
            imids = imids.cpu()

        feat = einops.rearrange(feat, "(b n) d -> b n d", b=bs)
        imids = einops.rearrange(imids, "(b n) -> b n", b=bs)
        return dict(out=feat, ids=imids)
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

