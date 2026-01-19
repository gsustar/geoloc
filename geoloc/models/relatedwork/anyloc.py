import torch
import einops


from .anyloclike import AnyLocLikeModel
from ..aggregators.vlad import VLAD

class AnyLoc(AnyLocLikeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.aggregator, VLAD)
        self.vlad = self.aggregator
        self.automatic_optimization = False

    def forward(self, x, return_residuals=False):
        out = dict()
        if self.rotator is not None:
            x, theta = self.rotator(x).values()
            out["theta"] = theta
        x = self.backbone(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        if return_residuals:
            qry_residuals = self.vlad.generate_multi_res_vec(x)
            out["qry_residuals"] = qry_residuals
        x = self.vlad.generate_multi(x)
        x = (
            self.pca.transform(x.detach().cpu().numpy())
            if self.pca is not None
            else x.cpu().numpy()
        )
        x = torch.from_numpy(x).float()
        out["out"] = x
        return out