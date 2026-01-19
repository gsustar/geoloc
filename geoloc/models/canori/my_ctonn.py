import torch
import kornia
import inspect
import lightning as L
import torch.nn.functional as F

from ...utils import tensor2wandbimg


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

        if batch_idx == 0 and self.trainer.current_epoch % 20 == 0:
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

        # theta_error = torch.abs(final_sat_orient - final_drn_orient).detach().cpu()
        theta_error = torch.abs(final_sat_orient - final_drn_orient).cpu()
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