import torch
import einops
import torchvision
import transformers
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from .utils import freeze, unfreeze_layers, dino_processor, remove_registers_and_cls_token


class MultiScaleOutputMixin:
    """Mixin for models that support multi-scale feature extraction from hidden layers."""
    def setup_multiscale_output(
        self,
        backbone_num_hidden_layers=None,
        backbone_hidden_size=None,
        out_indices=None,
        out_channels=None,
    ):
        self.output_hidden_states = False
        self.out = nn.Identity()
        self.output_dim = backbone_hidden_size
        
        if out_indices is not None:
            if out_indices == "auto":
                out_indices = torch.arange(
                    backbone_num_hidden_layers - 1,
                    0,
                    -backbone_num_hidden_layers // 4
                ).tolist()[::-1]
            
            assert max(out_indices) <= backbone_num_hidden_layers - 1, \
                f"Max out_index {max(out_indices)} exceeds num_hidden_layers {backbone_num_hidden_layers}"
            assert out_channels is not None, \
                "out_channels must be specified when using out_indices"
            
            self.output_hidden_states = True
            in_channels = len(out_indices) * backbone_hidden_size
            self.out_indices = out_indices
            self.output_dim = out_channels
            
            self.out = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=True),
                nn.LayerNorm(out_channels)
            )


class AnyLocDINOv3Backbone(nn.Module):
    """Just a wrapper for HFaceDINOv3Backbone but with the additional option of returning the `value` facet"""

    def __init__(
        self,
        layer=-1,
        facet="value",
        use_cls=False,
        norm_descs=True,
        pretrained_model_name_or_path="facebook/dinov3-vitb16-pretrain-lvd1689m",
    ):
        super().__init__()
        self.layer = layer
        self.facet = facet
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        assert self.facet in ["value", "token"]

        self.backbone = HFaceDINOBackbone(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.model_name = pretrained_model_name_or_path
        self.backbone = freeze(self.backbone)
        self.backbone.eval()

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


class HFaceDINOBackbone(nn.Module, MultiScaleOutputMixin):
    def __init__(
        self, 
        pretrained_model_name_or_path="facebook/dinov3-vitl16-pretrain-sat493m",
        return_cls_token=False,
        num_trainable_blocks=0,
        out_indices=None,
        out_channels=None,
    ):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_name = pretrained_model_name_or_path.split("/")[1]
        self.backbone = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path
        )
        self.backbone = freeze(self.backbone)
        self.backbone = unfreeze_layers(self.backbone, num_trainable_blocks)
        # self.backbone.eval()
        self.output_dim = self.backbone.config.hidden_size
        self.patch_size = self.backbone.config.patch_size
        self.return_cls_token = return_cls_token
        self.num_trainable_blocks = num_trainable_blocks

        self.setup_multiscale_output(
            backbone_hidden_size=self.backbone.config.hidden_size,
            backbone_num_hidden_layers=self.backbone.config.num_hidden_layers,
            out_indices=out_indices,
            out_channels=out_channels,
        )

    def processor(self, x: torch.Tensor):
        return dino_processor(x, self.patch_size, return_latent_size=True)

    def forward(self, x: torch.Tensor):
        x, latent_size = self.processor(x)
        x = self.backbone(x, output_hidden_states=self.output_hidden_states)
        if self.output_hidden_states:
            x = torch.cat([x.hidden_states[index+1] for index in self.out_indices], dim=-1)
        else:
            x = x.last_hidden_state
        x = self.out(x)

        x, cls_token = remove_registers_and_cls_token(x, latent_size, return_cls_tokens=True)
        x = einops.rearrange(
            x, "b (h w) c -> b c h w", h=latent_size[0], w=latent_size[1]
        )
        if self.return_cls_token:
            return x, cls_token
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
        self.model_name = model_name

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


class RADIOBackbone(nn.Module, MultiScaleOutputMixin):
    def __init__(self,
            model_name="radio_v2.5-l",
            return_cls_token=False,
            out_indices=None,
            out_channels=None,
        ):
        super().__init__()
        self.model_name = model_name
        self.backbone = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_name, progress=False, skip_validation=True)
        self.backbone = freeze(self.backbone)
        self.return_cls_token = return_cls_token
        # self.backbone = unfreeze_layers(self.backbone, num_trainable_blocks)
        self.setup_multiscale_output(
            backbone_hidden_size=self.backbone.model.embed_dim,
            backbone_num_hidden_layers=len(self.backbone.model.blocks),
            out_indices=out_indices,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor):
        (summary, final), features = self.backbone.forward_intermediates(x, indices=self.out_indices)
        _, _, lh, lw = final.shape
        if self.output_hidden_states:
            x = torch.cat(features, dim=-3)
            x = einops.rearrange(x, "b c h w -> b (h w) c")
        else:
            x = final
        x = self.out(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=lh, w=lw)
        if self.return_cls_token:
            return x, summary
        return x


class CTONN_VGGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model_dic = self.VGG16_initializator()

        self.CBR1_ENC = self.make_layers_from_names(["conv1_1", "conv1_2"], model_dic, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR2_ENC = self.make_layers_from_names(["conv2_1", "conv2_2"], model_dic, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR3_ENC = self.make_layers_from_names(["conv3_1", "conv3_2", "conv3_3"], model_dic, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR4_ENC = self.make_layers_from_names(["conv4_1", "conv4_2", "conv4_3"], model_dic, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR5_ENC = self.make_layers_from_names(["conv5_1", "conv5_2", "conv5_3"], model_dic, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.output_dim = 512  # assuming input images are resized to 224x224

    def VGG16_initializator(self):
        layer_names = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3",
                    "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"]
        layers = list(torchvision.models.vgg16_bn(pretrained=True).features.children())
        layers = [x for x in layers if isinstance(x, nn.Conv2d)]
        layer_dic = dict(zip(layer_names, layers))
        return layer_dic


    def make_layers_from_names(self, names, model_dic, bn_dim, existing_layer=None):
        layers = []
        if existing_layer is not None:
            layers = [existing_layer, nn.BatchNorm2d(bn_dim, momentum=0.1), nn.ReLU(inplace=True)]
        for name in names:
            layers += [deepcopy(model_dic[name]), nn.BatchNorm2d(bn_dim, momentum=0.1), nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)
    
    # @torch.no_grad()
    def processor(self, x):
        # x = torchvision.transforms.functional.center_crop(x, 512)
        x = torchvision.transforms.functional.resize(x, (224, 224))
        x = torchvision.transforms.functional.normalize(
            x,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return x

    def forward(self, x):
        x = self.processor(x)
        xs = self.pool1(self.CBR1_ENC(x))
        xs = self.pool2(self.CBR2_ENC(xs))
        xs = self.pool3(self.CBR3_ENC(xs))
        xs = self.pool4(self.CBR4_ENC(xs))
        xs = self.pool5(self.CBR5_ENC(xs))
        # xs = xs.view(-1, 512 * 7 * 7)
        return xs