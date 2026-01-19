import os
import torch
import pickle
import einops
from ..vprmodel import VPRModel
from ...utils import DEBUG

class AnyLocLikeModel(VPRModel):
    def __init__(self, fit_step=1, **kwargs):
        super().__init__(**kwargs)
        self.fit_step = fit_step
        self._is_fitted = False
        self.automatic_optimization = False

    def on_train_epoch_start(self):
        self.features = []
        if self.vlad.c_centers is not None and self.pca is None:
            self._is_fitted = True
            # self.trainer.should_stop = True
            print("There is no need to fit with pre-fitted VLAD and no PCA.")

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    @torch.no_grad()
    def training_step(self, batch, batch_idx):
        if self._is_fitted:
            return
        if DEBUG > 1 and batch_idx > 5:
            return
        if batch_idx % self.fit_step != 0:
            return
        x = self.backbone(batch["image"])
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        self.features.append(x.detach())

    @torch.no_grad()
    def _fit(self):
        if self._is_fitted:
            return
        self.features = torch.vstack(self.features)
        if self.vlad.c_centers is None:
            print("Fitting VLAD features...")
            self.features = self.vlad.fit_and_generate(self.features)
        else:
            self.features = self.vlad.generate_multi(self.features)

        if self.pca is not None:
            print("Fitting PCA...")
            self.features = self.pca.fit_transform(self.features.cpu())
        del self.features

    def on_validation_epoch_start(self):
        if not self._is_fitted:
            self._fit()
            self._is_fitted = True
        return super().on_validation_epoch_start()

    def on_train_epoch_end(self):
        if not self._is_fitted:
            self._fit()
            self._is_fitted = True

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["vlad"] = self.vlad
        checkpoint["pca"] = self.pca

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.vlad = checkpoint["vlad"]
        self.pca = checkpoint["pca"]

    def load_from_legacy_checkpoint(self, checkpoint_path, **kwargs):
        assert os.path.isdir(checkpoint_path), "Legacy checkpoint for anyloc-like models should be a directory"
        self.load_vlad(checkpoint_path)
        self.load_pca(checkpoint_path)
        return self

    def load_vlad(self, loaddir):
        self.vlad.load_c_centers(os.path.join(loaddir, "c_centers.pt"))

    def load_pca(self, loaddir):
        pca_loadpath = os.path.join(loaddir, "pca.pkl")
        if os.path.exists(pca_loadpath):
            print("Loading PCA from file...")
            with open(pca_loadpath, "rb") as f:
                self.pca = pickle.load(f)
