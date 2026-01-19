import lightning as L
from ..aggregators.mixvpr import MixVPR
from ..backbones import MixVPRResNetBackbone
from ..vprmodel import VPRModel
import torch


class MixVPRModel(VPRModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def load_from_legacy_checkpoint(self, checkpoint_path):
    #     self.load_state_dict(torch.load(checkpoint_path), strict=True)