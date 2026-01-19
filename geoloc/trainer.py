import lightning as L


class GeolocTrainer(L.Trainer):
    def __init__(self, *args, train_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_config = train_config

    def fit(self, model, train_dataloaders = None, val_dataloaders = None, build_dataloader=None, datamodule = None, ckpt_path = None, weights_only = None):
        self.steps_per_epoch = len(train_dataloaders)
        self.build_dataloader = build_dataloader
        return super().fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path, weights_only)