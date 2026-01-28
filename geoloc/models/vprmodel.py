import torch
import lightning as L
import inspect

from lightning.pytorch.utilities import grad_norm

from ..config_parser import class_from_config
from ..pipeline.benchmark import benchmark_single, get_dataset_type, get_model_type
from ..utils import DEBUG

class VPRModel(L.LightningModule):
    def __init__(
        self,
        backbone,
        aggregator,
        rotator=None,
        # segmentor=None,
        pca=None,
        optimizer=None,
        scheduler=None,
        loss=None,
        miner=None,
        **kwargs,
    ):
        super().__init__()
        self.rotator = rotator
        # self.segmentor = segmentor
        self.backbone = backbone
        self.aggregator = aggregator
        self.pca = pca
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss
        self.miner = miner
        
        # self.save_hyperparameters(ignore=["backbone", "segmentor"])  # write hyperparams into a file
        self.batch_acc = (
            []
        )  # we will keep track of the % of trivial pairs/triplets at the loss level

    def forward(self, x, **kwargs):
        BS, ch, h, w = x.shape
        out = dict()
        if self.rotator is not None:
            x, theta = self.rotator(x).values()
            out["theta"] = theta
        x = self.backbone(x)
        return_salad_matrix = kwargs.get("return_salad_matrix", False)
        agg_out = self.aggregator(x, return_salad_matrix=return_salad_matrix)
        if isinstance(agg_out, tuple):
            x, salad_matrix = agg_out
            out["salad_matrix"] = salad_matrix
        else:
            x = agg_out
        out["out"] = x
        return out

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

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def loss_function(self, descriptors, labels):
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                # somes losses do the online mining inside (they don't need a miner objet),
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class,
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss
    
    def _unpack_training_batch(self, batch):
        imgs = batch["images"]
        # if imgs.ndim == 4:
        #     imgs = imgs.unsqueeze(1)  # add num_same_place dim
        BS, N, ch, h, w = imgs.shape
        imgs = imgs.view(BS * N, ch, h, w)
        labels = torch.arange(BS).repeat_interleave(N)
        return imgs, labels
    
    def train_forward(self, x):
        return self.forward(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = self._unpack_training_batch(batch)
        descriptors = self.train_forward(imgs)["out"]
        # descriptors = self(imgs)["out"]

        if torch.isnan(descriptors).any():
            raise ValueError("NaNs in descriptors")

        loss = self.loss_function(descriptors, labels)
        self.log("loss", loss.item(), logger=True, prog_bar=True)
        self.extra_logs()
        return {"loss": loss}

    def extra_logs(self):
        if hasattr(self.backbone, "logging_info"):
            for k, v in self.backbone.logging_info.items():
                self.log(f"backbone/{k}", v, logger=True, prog_bar=False)

    def on_train_epoch_end(self):
        self.batch_acc = []

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["my_config"] = self.my_config

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.my_config = checkpoint["my_config"]

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        weights_only=None,
        **kwargs
    ):
        try:
            return super(VPRModel, cls).load_from_checkpoint(
                checkpoint_path, map_location=map_location, 
                hparams_file=hparams_file, strict=strict, 
                weights_only=weights_only, **kwargs
            )
        except:
            obj = cls(**kwargs)
            return obj.load_from_legacy_checkpoint(checkpoint_path, **kwargs)

    def load_from_legacy_checkpoint(self, checkpoint_path, **kwargs):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        self.load_state_dict(checkpoint, strict=False)
        return self

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        keys_to_remove = []
        for name, param in self.named_parameters():
            if (param.requires_grad is False 
                or name.startswith("segmentor.") 
                or name.startswith("backbone.")
            ):
                if not param.requires_grad:
                    keys_to_remove.append(name)
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]
        return state_dict

    def on_validation_epoch_start(self):
        self.val_benchmark_top_k = self.trainer.train_config.benchmark_config.benchmark_top_k
        self.val_recalls_at_k = torch.zeros(len(self.val_benchmark_top_k))
        self.num_qry_images = 0
        print("Building database...")
        self.vdb = class_from_config(self.trainer.train_config.build_config.vdb)
        self.vdb.build(
            self.trainer.build_dataloader,
            model=self,
            rotation_angles=None,
            device=self.device,
            verbose=True,
        )

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        qry_image_dataset = self.trainer.val_dataloaders.dataset
        ref_image_dataset = self.trainer.build_dataloader.dataset
        benchmark_results = benchmark_single(
            model=self, model_type=get_model_type(self),
            vdb=self.vdb, vdbdir=self.vdb.vdbdir,
            qry_image_dataset=qry_image_dataset,
            ref_image_dataset=ref_image_dataset,
            benchmark_top_k=self.val_benchmark_top_k,
            dataset_type=get_dataset_type(qry_image_dataset),
            query_ix=batch_idx,
            device=self.device,
        )
        self.num_qry_images += 1
        recall_at_k = benchmark_results["recall_at_k"]
        intersection_recall_at_k = benchmark_results["intersection_recall_at_k"]
        self.val_recalls_at_k += torch.from_numpy(
            recall_at_k if intersection_recall_at_k is None else intersection_recall_at_k
        )

    def on_validation_epoch_end(self):
        self.val_recalls_at_k = self.val_recalls_at_k / self.num_qry_images
        for i, k in enumerate(self.val_benchmark_top_k):
            self.log(f'val/R{k}', self.val_recalls_at_k[i], prog_bar=False, logger=True)
        del self.val_recalls_at_k
        del self.vdb
