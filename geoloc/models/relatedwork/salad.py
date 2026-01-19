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
    
    # def load_from_legacy_checkpoint(self, checkpoint_path):
    #     self.load_state_dict(torch.load(checkpoint_path), strict=True)