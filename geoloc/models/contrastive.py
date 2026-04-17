import torch
import kornia
import torch.nn.functional as F
from .vprmodel import VPRModel

class ContrastiveModel(VPRModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _unpack_training_batch(self, batch):
        imgs = batch["images"]
        return imgs, None

    def train_forward(self, x):
        BS, N, ch, h, w = x.shape
        embeddings = []
        # NOTE: For loop to avoid out of memory, since the backbone may be too large to process all images at once
        for i in range(BS):
            embed = self.backbone(x[i])
            embed = self.aggregator(embed)
            embeddings.append(embed)
        embeddings = torch.stack(embeddings, dim=0)
        return dict(out=embeddings)

    def loss_function(self, descriptors, labels=None):
        loss = self.loss_fn(descriptors)
        return loss


class STContrastiveModel(ContrastiveModel):
    def __init__(self, *args, st_loss, training_step_mode="normal", **kwargs):
        super().__init__(*args, **kwargs)
        self.st_loss = st_loss
        self.training_step_mode = training_step_mode

    def train_forward(self, x):
        BS, N, ch, h, w = x.shape
        features = []
        embeddings = []
        for i in range(BS):
            embed = self.backbone(x[i])
            feats = embed[0] if isinstance(embed, tuple) else embed
            features.append(feats)
            embed = self.aggregator(embed)
            embeddings.append(embed)
        features = torch.stack(features, dim=0)
        embeddings = torch.stack(embeddings, dim=0)
        return dict(out=embeddings, features=features)
    
    # def __training_step_unrotate_features(self, batch, batch_idx):
    def training_step_rotating(self, batch, batch_idx):
        imgs, labels = self._unpack_training_batch(batch)
        # !!! make sure that the images are north aligned
        assert imgs.ndim == 5 and imgs.shape[1] == 2, "Currently only support input images of shape (B, 2, C, H, W)"
        drn_imgs = imgs[:, 0, :, :, :]  # (B, C, H, W)
        sat_imgs = imgs[:, 1, :, :, :]  # (B, C, H, W)

        # Randomly rotated satellite images by alpha0
        # alpha0 = torch.randint(0, self.k, (sat_imgs.size(0),)).to(self.device) * (2 * torch.pi / self.k)
        alpha0 = torch.randn(sat_imgs.size(0), device=self.device) * (2 * torch.pi)
        sat_alpha0 = kornia.geometry.transform.rotate(sat_imgs, alpha0 * 180.0 / torch.pi)
        drn_alpha0 = kornia.geometry.transform.rotate(drn_imgs, alpha0 * 180.0 / torch.pi)

        # Randomly rotated drone images and satellite images by alpha1
        # alpha1 = torch.randint(0, self.k, (sat_imgs.size(0),)).to(self.device) * (2 * torch.pi / self.k)
        alpha1 = torch.randn(sat_imgs.size(0), device=self.device) * (2 * torch.pi)
        sat_alpha1 = kornia.geometry.transform.rotate(sat_imgs, alpha1 * 180.0 / torch.pi)
        drn_alpha1 = kornia.geometry.transform.rotate(drn_imgs, alpha1 * 180.0 / torch.pi)

        in_imgs = torch.stack([sat_alpha0, sat_alpha1, drn_alpha0, drn_alpha1], dim=1)
        output = self.train_forward(in_imgs)
        embs = output['out']
        feats = output['features']
        
        sat_alpha0_emb, sat_alpha1_emb, drn_alpha0_emb, drn_alpha1_emb = map(lambda lst: lst.squeeze(1), torch.chunk(embs, 4, dim=1))
        sat_alpha0_feat, sat_alpha1_feat, drn_alpha0_feat, drn_alpha1_feat = map(lambda lst: lst.squeeze(1), torch.chunk(feats, 4, dim=1))

        # reverse the rotations on features
        sat_alpha0_feat_unrot = kornia.geometry.transform.rotate(sat_alpha0_feat, -alpha0 * 180.0 / torch.pi)
        sat_alpha1_feat_unrot = kornia.geometry.transform.rotate(sat_alpha1_feat, -alpha1 * 180.0 / torch.pi)
        drn_alpha0_feat_unrot = kornia.geometry.transform.rotate(drn_alpha0_feat, -alpha0 * 180.0 / torch.pi)
        drn_alpha1_feat_unrot = kornia.geometry.transform.rotate(drn_alpha1_feat, -alpha1 * 180.0 / torch.pi)


        feats = torch.stack([sat_alpha0_feat_unrot, sat_alpha1_feat_unrot, drn_alpha0_feat_unrot, drn_alpha1_feat_unrot], dim=1)
        embs = torch.stack([sat_alpha0_emb, sat_alpha1_emb, drn_alpha0_emb, drn_alpha1_emb], dim=1)

        if torch.isnan(embs).any():
            raise ValueError("NaNs in descriptors")

        loss = self.loss_function(embs, feats, labels)
        self.extra_logs()
        return {"loss": loss}

    def training_step_normal(self, batch, batch_idx):
        imgs, labels = self._unpack_training_batch(batch)
        assert imgs.ndim == 5 and imgs.shape[1] == 2, "Currently only support input images of shape (B, 2, C, H, W)"
        output = self.train_forward(imgs)
        embs = output['out']
        feats = output['features']
        
        if torch.isnan(embs).any():
            raise ValueError("NaNs in descriptors")

        loss = self.loss_function(embs, feats, labels)
        self.extra_logs()
        return {"loss": loss}
    
    def training_step(self, batch, batch_idx):
        if self.training_step_mode == "rotating":
            return self.training_step_rotating(batch, batch_idx)
        else:
            return self.training_step_normal(batch, batch_idx)
    
    def loss_function(self, descriptors, features=None, labels=None):
        emb_loss = self.loss_fn(descriptors)
        ft_loss = self.st_loss(features)
        loss = emb_loss + ft_loss
        self.log("emb_loss", emb_loss.item(), logger=True, prog_bar=True)
        self.log("ft_loss", ft_loss.item(), logger=True, prog_bar=True)
        self.log("loss", loss.item(), logger=True, prog_bar=True)
        return loss