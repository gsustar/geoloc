import lightning as L
import torch

from torch.optim import lr_scheduler

from ..aggregators.salad import SALAD
from ..backbones import SaladDINOv2Backbone
from ...config_parser import dict_to_namespace
from ..vprmodel import VPRModel


class SALADModel(VPRModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def forward(self, x, return_salad_matrix=False):
    #     BS, ch, h, w = x.shape
    #     out = dict()
    #     if self.rotator is not None:
    #         x, theta = self.rotator(x).values()
    #         out["theta"] = theta
    #     x = self.backbone(x)
    #     if return_salad_matrix:
    #         x, salad_matrix = self.aggregator(x, return_salad_matrix=return_salad_matrix)
    #         out["salad_matrix"] = salad_matrix
    #     else:
    #         x = self.aggregator(x)
    #     out["out"] = x
    #     return out
    
    # def load_from_legacy_checkpoint(self, checkpoint_path):
    #     self.load_state_dict(torch.load(checkpoint_path), strict=True)