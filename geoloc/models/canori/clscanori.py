import torch
import kornia
import numpy as np
import torch.nn as nn
import lightning as L
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=-3),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class ClsCanori(L.LightningModule):
    def __init__(
        self,
        backbone,
        optimizer,
        scheduler=None,
        k=36,
    ):
        super().__init__()
        self.k = k
        self.backbone = backbone
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.classification_head = ClassificationHead(
            in_channels=self.backbone.output_dim,
            hidden_channels=4096,
            num_classes=self.k,
        )

    def get_theta_from_cls(self, theta_cls):
        theta = theta_cls.float() * (2 * torch.pi / self.k)
        return theta

    def forward(self, x):
        xs = self.backbone(x)
        xs = self.classification_head(xs)
        theta_cls = torch.argmax(xs, dim=1)
        theta = self.get_theta_from_cls(theta_cls)

        rot_x = kornia.geometry.transform.rotate(
            x,
            angle=theta.squeeze().float() * 180.0 / torch.pi
        )
        return dict(rot_x=rot_x, theta=theta)


    # def training_step(self, batch, batch_idx):
    #     drn_imgs = batch["images"][:, 0, :, :, :]  # (B, C, H, W)
    #     sat_imgs = batch["images"][:, 1, :, :, :]  # (B, C, H, W)

    #     # Randomly rotated satellite images by alpha0
    #     alpha0 = torch.randint(0, self.k, (sat_imgs.size(0),)).to(self.device) * (2 * torch.pi / self.k)
    #     sat_alpha0 = kornia.geometry.transform.rotate(sat_imgs, alpha0 * 180.0 / torch.pi)

    #     # Randomly rotated drone images and satellite images by alpha1
    #     alpha1 = torch.randint(0, self.k, (sat_imgs.size(0),)).to(self.device) * (2 * torch.pi / self.k)
    #     # sat_alpha_1 = kornia.geometry.transform.rotate(sat_imgs, alpha1 * 180.0 / torch.pi)
    #     drn_alpha1 = kornia.geometry.transform.rotate(drn_imgs, alpha1 * 180.0 / torch.pi)

    #     output0 = 
    #     output0, output1 = self.stn_model(sat_alpha0, alpha0, drn_alpha1, alpha1)

    #     feat0, theta0 = output0
    #     feat1, theta1 = output1

    #     loss = self.loss_function(feat0, theta0, alpha0, feat1, theta1, alpha1, split="train")

    #     if batch_idx == 0 and self.trainer.current_epoch % 20 == 0:
    #         rot_sat_imgs = kornia.geometry.transform.rotate(
    #             sat_alpha0,
    #             angle=theta0.squeeze().float() * 180.0 / torch.pi,
    #         )
    #         rot_drn_imgs = kornia.geometry.transform.rotate(
    #             drn_alpha1,
    #             angle=theta1.squeeze().float() * 180.0 / torch.pi,
    #         )

    #         self.logger.experiment.log({
    #             "train/sat_images": tensor2wandbimg(sat_imgs),
    #             "train/drn_images": tensor2wandbimg(drn_imgs),
    #             "train/sat_alpha0_images": tensor2wandbimg(sat_alpha0),
    #             "train/drn_alpha1_images": tensor2wandbimg(drn_alpha1),
    #             "train/rotated_sat_images": tensor2wandbimg(rot_sat_imgs),
    #             "train/rotated_drn_images": tensor2wandbimg(rot_drn_imgs),
    #         })
    #     return loss