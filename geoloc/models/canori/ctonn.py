import torch
import lightning as L
import kornia
import torch.nn as nn
import torch.nn.functional as F
import inspect

from copy import deepcopy
import math
from torch.autograd import Variable
from torchvision import models
import torchvision

from ...utils import tensor2wandbimg


def VGG16_initializator():
    layer_names = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2", "conv3_3",
                   "conv4_1", "conv4_2", "conv4_3", "conv5_1", "conv5_2", "conv5_3"]
    layers = list(models.vgg16_bn(pretrained=True).features.children())
    layers = [x for x in layers if isinstance(x, nn.Conv2d)]
    layer_dic = dict(zip(layer_names, layers))
    return layer_dic


def make_layers_from_names(names, model_dic, bn_dim, existing_layer=None):
    layers = []
    if existing_layer is not None:
        layers = [existing_layer, nn.BatchNorm2d(bn_dim, momentum=0.1), nn.ReLU(inplace=True)]
    for name in names:
        layers += [deepcopy(model_dic[name]), nn.BatchNorm2d(bn_dim, momentum=0.1), nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)


N_PARAMS = {'affine': 6,
            'rotation': 1}


# Spatial transformer network forward function
def stn(x, theta, mode='rotation', reduce_ratio=28/224):
    rr = reduce_ratio
    if mode == 'affine':
        theta1 = theta.view(-1, 2, 3)
    else:
        theta1 = Variable(torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device()), requires_grad=True)
        theta1 = theta1 + 0
        theta1[:, 0, 0] = 1.0
        theta1[:, 1, 1] = 1.0
        if mode == 'rotation':
            angle = theta[:, 0]
            theta1[:, 0, 0] = torch.cos(angle) * rr
            theta1[:, 0, 1] = -torch.sin(angle) * rr
            theta1[:, 1, 0] = torch.sin(angle) * rr
            theta1[:, 1, 1] = torch.cos(angle) * rr

    target_size = [x.size(0), x.size(1), 28, 28]
    # target_size = [x.size(0), x.size(1), int(reduce_ratio * 244), int(reduce_ratio * 244)]
    grid = F.affine_grid(theta1, target_size, align_corners=True)
    x = F.grid_sample(x, grid, align_corners=True)
    return x

# def stn(x, theta, mode='rotation', reduce_ratio=28/224):
#     rr = reduce_ratio
#     if mode == 'affine':
#         theta1 = theta.view(-1, 2, 3)
#     else:
#         # theta1 = Variable(torch.zeros([x.size(0), 2, 3], dtype=torch.float32, device=x.get_device()), requires_grad=True)
#         # theta1 = theta1 + 0
#         # theta1[:, 0, 0] = 1.0
#         # theta1[:, 1, 1] = 1.0
#         # if mode == 'rotation':
#         #     angle = theta[:, 0]
#         #     theta1[:, 0, 0] = torch.cos(angle) * rr
#         #     theta1[:, 0, 1] = -torch.sin(angle) * rr
#         #     theta1[:, 1, 0] = torch.sin(angle) * rr
#         #     theta1[:, 1, 1] = torch.cos(angle) * rr
#         cos_t = torch.cos(theta)
#         sin_t = torch.sin(theta)
#         zeros = torch.zeros_like(cos_t)
#         theta1 = torch.stack(
#             [
#                 cos_t*rr, -sin_t*rr, zeros,
#                 sin_t*rr,  cos_t*rr, zeros
#             ],
#             dim=1
#         ).view(-1, 2, 3)

#     target_size = [x.size(0), x.size(1), 28, 28]
#     # target_size = [x.size(0), x.size(1), int(reduce_ratio * 244), int(reduce_ratio * 244)]
#     grid = F.affine_grid(theta1, target_size)
#     x = F.grid_sample(x, grid)
#     return x


class EmbeddingNet(nn.Module):
    def __init__(self, stn_mode='rotation'):
        super(EmbeddingNet, self).__init__()

        model_dic = VGG16_initializator()

        self.CBR1_ENC = make_layers_from_names(["conv1_1", "conv1_2"], model_dic, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR2_ENC = make_layers_from_names(["conv2_1", "conv2_2"], model_dic, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR3_ENC = make_layers_from_names(["conv3_1", "conv3_2", "conv3_3"], model_dic, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR4_ENC = make_layers_from_names(["conv4_1", "conv4_2", "conv4_3"], model_dic, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.CBR5_ENC = make_layers_from_names(["conv5_1", "conv5_2", "conv5_3"], model_dic, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stn_mode = stn_mode
        self.stn_n_params = N_PARAMS[stn_mode]

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            # nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.stn_n_params),
            nn.Tanh()
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[3].weight.data.fill_(0)
        self.fc_loc[3].weight.data.zero_()
        if self.stn_mode == 'affine':
            self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        elif self.stn_mode == 'rotation':
            self.fc_loc[3].bias.data.copy_(torch.tensor([0], dtype=torch.float))

    def processor(self, x):
        # x = torchvision.transforms.functional.center_crop(x, 512)
        x = torchvision.transforms.functional.resize(x, (224, 224))
        x = torchvision.transforms.functional.normalize(
            x,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return x


    def theta(self, x):
        x = self.processor(x)
        xs = self.pool1(self.CBR1_ENC(x))
        xs = self.pool2(self.CBR2_ENC(xs))
        xs = self.pool3(self.CBR3_ENC(xs))
        xs = self.pool4(self.CBR4_ENC(xs))
        xs = self.pool5(self.CBR5_ENC(xs))
        # xs = xs.view(-1, 512 * 8 * 8)
        xs = xs.view(-1, 512 * 7 * 7)
        theta = self.fc_loc(xs)  # for rotation: only 1 param, it is the angle
        theta = theta * math.pi * 1.5
        return theta

    def forward(self, x):
        # get the theta
        theta = self.theta(x)
        return theta


class StnNet2(nn.Module):
    def __init__(self, embedding_net):
        super(StnNet2, self).__init__()
        self.embedding_net = embedding_net

    def forward_single(self, x):
        # get the theta
        theta = self.embedding_net(x)
        # v = stn(x, theta, mode='rotation', reduce_ratio=224/224)
        v = stn(x, theta, mode='rotation', reduce_ratio=28/224)
        return v, theta

    def forward(self, x0, alpha0, x1, alpha1):
        # get the theta
        # theta0 = self.embedding_net(x0).view(-1)
        # theta1 = self.embedding_net(x1).view(-1)
        theta0 = self.embedding_net(x0)
        theta1 = self.embedding_net(x1)

        # v0 = stn(x0, theta0, mode='rotation', reduce_ratio=224/224)
        # v1 = stn(x0, (alpha1-alpha0+theta1), mode='rotation', reduce_ratio=224/224)
        v0 = stn(x0, theta0, mode='rotation', reduce_ratio=28/224)
        v1 = stn(x0, (alpha1-alpha0+theta1), mode='rotation', reduce_ratio=28/224)

        return (v0, theta0), (v1, theta1)


class LCTONN(L.LightningModule):
    def __init__(
        self,
        stn_model,
        optimizer,
        scheduler=None,
        losses = ["theta_mse", "feat_l1"],
        lmbd_losses = [1.0, 0.1],
        k=36,
    ):
        super().__init__()
        self.k = k
        self.stn_model = stn_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.losses = losses
        self.lmbd_losses = lmbd_losses

    def forward(self, x):
        _, theta = self.stn_model.forward_single(x)
        rot_x = kornia.geometry.transform.rotate(
            x,
            angle=theta.squeeze().float() * 180.0 / torch.pi
        )
        return dict(rot_x=rot_x, theta=theta)
        # return rot_x

    def training_step(self, batch, batch_idx):
        drn_imgs = batch["images"][:, 0, :, :, :]  # (B, C, H, W)
        sat_imgs = batch["images"][:, 1, :, :, :]  # (B, C, H, W)

        # Randomly rotated satellite images by alpha0
        alpha0 = torch.randint(0, self.k, (sat_imgs.size(0),)).to(self.device) * (2 * torch.pi / self.k)
        sat_alpha0 = kornia.geometry.transform.rotate(sat_imgs, alpha0 * 180.0 / torch.pi)

        # Randomly rotated drone images and satellite images by alpha1
        alpha1 = torch.randint(0, self.k, (sat_imgs.size(0),)).to(self.device) * (2 * torch.pi / self.k)
        # sat_alpha_1 = kornia.geometry.transform.rotate(sat_imgs, alpha1 * 180.0 / torch.pi)
        drn_alpha1 = kornia.geometry.transform.rotate(drn_imgs, alpha1 * 180.0 / torch.pi)

        output0, output1 = self.stn_model(sat_alpha0, alpha0, drn_alpha1, alpha1)

        feat0, theta0 = output0
        feat1, theta1 = output1

        loss = self.loss_function(feat0, theta0, alpha0, feat1, theta1, alpha1, split="train")

        if batch_idx == 0 and self.trainer.current_epoch % 10 == 0:
            rot_sat_imgs = kornia.geometry.transform.rotate(
                sat_alpha0,
                angle=theta0.squeeze().float() * 180.0 / torch.pi,
            )
            rot_drn_imgs = kornia.geometry.transform.rotate(
                drn_alpha1,
                angle=theta1.squeeze().float() * 180.0 / torch.pi,
            )

            self.logger.experiment.log({
                "train/sat_images": tensor2wandbimg(sat_imgs),
                "train/drn_images": tensor2wandbimg(drn_imgs),
                "train/sat_alpha0_images": tensor2wandbimg(sat_alpha0),
                "train/drn_alpha1_images": tensor2wandbimg(drn_alpha1),
                "train/rotated_sat_images": tensor2wandbimg(rot_sat_imgs),
                "train/rotated_drn_images": tensor2wandbimg(rot_drn_imgs),
            })
        return loss
    
    def validation_step(self, batch, batch_idx):
        drn_imgs = batch["images"][:, 0, :, :, :] # (B, C, H, W)
        sat_imgs = batch["images"][:, 1, :, :, :] # (B, C, H, W)
        drn_yaw = batch["yaw"]

        v0, theta_sat = self.stn_model.forward_single(sat_imgs)
        v1, theta_drn = self.stn_model.forward_single(drn_imgs)

        theta_drn = theta_drn.view(-1)
        theta_sat = theta_sat.view(-1)

        final_sat_orient = (0.0 + theta_sat) % (2 * torch.pi)
        final_drn_orient = (drn_yaw + theta_drn) % (2 * torch.pi)

        theta_error = torch.abs(final_sat_orient - final_drn_orient).detach().cpu()
        theta_error = torch.min(theta_error, 2 * torch.pi - theta_error)
        self.theta_errors.append(theta_error)
        # self.theta_errors.append(
        #     torch.abs((-drn_yaw + theta_drn) - (theta_sat + 0.0)).detach().cpu()
        # )

        self.log("val/theta_mean", torch.mean(torch.abs(theta_sat)), prog_bar=True, logger=True)

        if batch_idx % 100 == 0:
            rot_sat_imgs = kornia.geometry.transform.rotate(
                sat_imgs,
                angle=theta_sat.squeeze().float() * 180.0 / torch.pi,
            )
            rot_drn_imgs = kornia.geometry.transform.rotate(
                drn_imgs,
                angle=theta_drn.squeeze().float() * 180.0 / torch.pi,
            )
            self.vis_val_sat_imgs.append(sat_imgs.clone())
            self.vis_val_drn_imgs.append(drn_imgs.clone())
            self.vis_val_rot_sat_imgs.append(rot_sat_imgs.clone())
            self.vis_val_rot_drn_imgs.append(rot_drn_imgs.clone())

    def on_validation_epoch_start(self):
        self.theta_errors = []
        self.vis_val_sat_imgs = []
        self.vis_val_drn_imgs = []
        self.vis_val_rot_sat_imgs = []
        self.vis_val_rot_drn_imgs = []

    def on_validation_epoch_end(self):
        self.theta_errors = torch.cat(self.theta_errors, dim=0)
        mae = torch.mean(torch.abs(self.theta_errors))
        median = torch.median(torch.abs(self.theta_errors))
        self.log("val/theta_median", median, prog_bar=True, logger=True)
        self.log("val/theta_mae", mae, prog_bar=True, logger=True)

        self.vis_val_sat_imgs = torch.cat(self.vis_val_sat_imgs, dim=0)
        self.vis_val_drn_imgs = torch.cat(self.vis_val_drn_imgs, dim=0)
        self.vis_val_rot_drn_imgs = torch.cat(self.vis_val_rot_drn_imgs, dim=0)
        self.vis_val_rot_sat_imgs = torch.cat(self.vis_val_rot_sat_imgs, dim=0)

        self.logger.experiment.log({
            "val/sat_images": tensor2wandbimg(self.vis_val_sat_imgs, max_imgs=8),
            "val/drn_images": tensor2wandbimg(self.vis_val_drn_imgs, max_imgs=8),
            "val/rotated_sat_images": tensor2wandbimg(self.vis_val_rot_sat_imgs, max_imgs=8),
            "val/rotated_drn_images": tensor2wandbimg(self.vis_val_rot_drn_imgs, max_imgs=8),
        })

    def loss_function(self, feat0, theta0, alpha0, feat1, theta1, alpha1, split="train"):
        loss_dict = {}
        total_loss = 0.0
        for lmbd, loss in zip(self.lmbd_losses, self.losses):
            if loss == "theta_mse":
                curr_loss = F.mse_loss(alpha0 + theta0.view(-1), alpha1 + theta1.view(-1))
            elif loss == "theta_mae":
                curr_loss = F.l1_loss(alpha0 + theta0.view(-1), alpha1 + theta1.view(-1))
            elif loss == "feat_l1":
                curr_loss = F.l1_loss(feat0, feat1)
            else:
                raise NotImplementedError(f"Loss {loss} not implemented in loss_function")
            curr_loss = lmbd * curr_loss
            loss_dict[f"{split}/{loss}"] = curr_loss
            total_loss += curr_loss
        
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        self.log(f"{split}/total_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        if self.optimizer is None:
            return None
        self.optimizer = self.optimizer(params=self.parameters())
        retdict = {"optimizer": self.optimizer}
        if self.scheduler is not None:
            schduler_args = {}
            if "steps_per_epoch" in inspect.signature(self.scheduler).parameters:
                schduler_args["steps_per_epoch"] = self.trainer.steps_per_epoch
            self.scheduler = self.scheduler(optimizer=self.optimizer, **schduler_args)
            interval = self.my_config.init_args.scheduler.get("interval", "step")
            retdict.update(
                {"lr_scheduler": {"scheduler": self.scheduler, "interval": interval, "frequency": 1}}
            )
        return retdict