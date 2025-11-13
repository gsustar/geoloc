from typing_extensions import Literal
import torch
import transformers
import torch.nn as nn
import einops
import torchvision

from .utils import freeze
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F


def dino_processor(x: torch.Tensor, patch_size: int, return_latent_size: bool = False):
    c, h, w = x.shape[-3:]
    new_h, new_w = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    x = TF.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    x = TF.center_crop(x, (new_h, new_w))
    if return_latent_size:
        latent_size = (new_h // patch_size, new_w // patch_size)
        return x, latent_size
    return x


def remove_registers_and_cls_token(backbone_features: torch.Tensor, latent_size: tuple):
    num_feat_tkns = latent_size[0] * latent_size[1]
    num_all_tkns = backbone_features.shape[1]
    start_idx = num_all_tkns - num_feat_tkns
    return backbone_features[:, start_idx:, ...]


class AnyLocDINOv3Backbone(nn.Module):
    """Just a wrapper for HFaceDINOv3Backbone but with the additional option of returning the `value` facet"""

    def __init__(
        self,
        layer=-1,
        facet="value",
        use_cls=False,
        norm_descs=True,
    ):
        super().__init__()
        self.layer = layer
        self.facet = facet
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        assert self.facet in ["value", "token"]

        self.backbone = HFaceDINOv3Backbone()

        if self.facet == "token":
            self.fn_handle = self.backbone.backbone.layer[
                self.layer
            ].register_forward_hook(self._generate_forward_hook())
        else:
            self.fn_handle = self.backbone.backbone.layer[
                self.layer
            ].attention.v_proj.register_forward_hook(self._generate_forward_hook())
        self._hook_out = None

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output

        return _forward_hook

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            res = self.backbone(x)
            latent_size = res.shape[-2:]
            if self.use_cls:
                res = self._hook_out
            else:
                res = remove_registers_and_cls_token(self._hook_out, latent_size)
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None  # Reset the hook
        res = einops.rearrange(
            res, "b (h w) c -> b c h w", h=latent_size[0], w=latent_size[1]
        )
        return res

    def __del__(self):
        self.fn_handle.remove()


class AnyLocDINOv2Backbone(nn.Module):
    def __init__(
        self,
        model_name="dinov2_vitg14",
        layer=-1,
        facet="value",
        use_cls=False,
        norm_descs=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        self.facet = facet
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.patch_size = 14

        assert self.facet in ["query", "key", "value", "token"]

        self.backbone = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.backbone = freeze(self.backbone)
        self.backbone.eval()

        if self.facet == "token":
            self.fh_handle = self.backbone.blocks[self.layer].register_forward_hook(
                self._generate_forward_hook()
            )
        else:
            self.fh_handle = self.backbone.blocks[
                self.layer
            ].attn.qkv.register_forward_hook(self._generate_forward_hook())

        self._hook_out = None

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output

        return _forward_hook

    def processor(self, x: torch.Tensor):
        return dino_processor(x, self.patch_size, return_latent_size=True)

    def forward(self, x: torch.Tensor):
        x, latent_size = self.processor(x)
        with torch.no_grad():
            x = self.backbone(x)
            if self.use_cls:
                res = self._hook_out
            else:
                res = remove_registers_and_cls_token(self._hook_out, latent_size)
                # res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len : 2 * d_len]
                else:
                    res = res[:, :, 2 * d_len :]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None  # Reset the hook
        res = einops.rearrange(
            res, "b (h w) c -> b c h w", h=latent_size[0], w=latent_size[1]
        )
        return res

    def __del__(self):
        self.fh_handle.remove()


class HFaceDINOv2Backbone(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path="facebook/dinov2-giant",
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.backbone = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.backbone = freeze(self.backbone)
        self.backbone.eval()
        self.output_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size

    def processor(self, x: torch.Tensor):
        return dino_processor(x, self.patch_size, return_latent_size=True)

    def forward(self, x: torch.Tensor):
        x, latent_size = self.processor(x)
        x = self.backbone(x)
        x = remove_registers_and_cls_token(x.last_hidden_state, latent_size)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, latent_size[0], latent_size[1])
        return x


class HFaceDINOv3Backbone(nn.Module):
    def __init__(
        self, pretrained_model_name_or_path="facebook/dinov3-vitl16-pretrain-sat493m"
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.backbone = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.backbone = freeze(self.backbone)
        self.backbone.eval()
        self.output_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size

    def processor(self, x: torch.Tensor):
        return dino_processor(x, self.patch_size, return_latent_size=True)

    def forward(self, x: torch.Tensor):
        x, latent_size = self.processor(x)
        x = self.backbone(x)
        x = remove_registers_and_cls_token(x.last_hidden_state, latent_size)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, latent_size[0], latent_size[1])
        return x


DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class SaladDINOv2Backbone(nn.Module):
    """
    DINOv2 model

    Args:
            model_name (str): The name of the model architecture
                    should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
            num_trainable_blocks (int): The number of last blocks in the model that are trainable.
            norm_layer (bool): If True, a normalization layer is applied in the forward pass.
            return_token (bool): If True, the forward pass returns both the feature map and the token.
    """

    def __init__(
        self,
        model_name="dinov2_vitb14",
        num_trainable_blocks=2,
        norm_layer=False,
        return_token=False,
    ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f"Unknown model name {model_name}"
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
                x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
                f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
                t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        x = dino_processor(x, patch_size=14, return_latent_size=False)

        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[: -self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks :]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f


class MixVPRResNetBackbone(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[],
    ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
                model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
                pretrained (bool, optional): Whether pretrained or not. Defaults to True.
                layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
                layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
                NotImplementedError: if the model_name corresponds to an unknown architecture.
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = "IMAGENET1K_V1"
        else:
            weights = None

        if "swsl" in model_name or "ssl" in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                "facebookresearch/semi-supervised-ImageNet1K-models", model_name
            )
        else:
            if "resnext50" in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif "resnet50" in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif "101" in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif "152" in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif "34" in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif "18" in model_name:
                # self.model = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif "wide_resnet50_2" in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError("Backbone architecture not recognized!")

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if "34" in model_name or "18" in model_name:
            out_channels = 512

        self.out_channels = (
            out_channels // 2 if self.model.layer4 is None else out_channels
        )
        self.out_channels = (
            self.out_channels // 2 if self.model.layer3 is None else self.out_channels
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x
