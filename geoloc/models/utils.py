import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


def freeze(model: nn.Module):
    """Freeze the model parameters."""
    for param in model.parameters():
        param.requires_grad = False
    return model

def unfreeze_layers(model: nn.Module, num_layers: int):
    """Unfreeze the last `num_layers` layers of the model."""
    # assert model.training is True, "Model must be in training mode to unfreeze layers."
    if num_layers <= 0:
        return model
    model.train()
    layers = None
    for name, module in model.named_modules():
        if name.endswith("layer"):
            layers = module; break
    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    return model

def dino_processor(x: torch.Tensor, patch_size: int, return_latent_size: bool = False):
    c, h, w = x.shape[-3:]
    new_h, new_w = (h // patch_size) * patch_size, (w // patch_size) * patch_size
    x = TF.normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    x = TF.center_crop(x, (new_h, new_w))
    if return_latent_size:
        latent_size = (new_h // patch_size, new_w // patch_size)
        return x, latent_size
    return x


def remove_registers_and_cls_token(backbone_features: torch.Tensor, latent_size: tuple, return_cls_tokens: bool = False):
    num_feat_tkns = latent_size[0] * latent_size[1]
    num_all_tkns = backbone_features.shape[1]
    start_idx = num_all_tkns - num_feat_tkns
    
    feature_tokens = backbone_features[:, start_idx:, ...]
    cls_token = backbone_features[:, 0, ...]
    registers = backbone_features[:, 1:start_idx, ...]
    if return_cls_tokens:
        return feature_tokens, cls_token
    
    return feature_tokens
