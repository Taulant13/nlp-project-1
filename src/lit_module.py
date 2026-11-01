"""Lightning wrapper shared by both architectures: loss, metrics, optimizer."""

import math

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm


class ArtModule(L.LightningModule):
    """Trains a model that maps a batch to logits of shape (B, 2)."""

    def __init__(self, model: torch.nn.Module, config: dict):
        super().__init__()
        # Stored in the checkpoint only; the logger already receives the config directly.
        self.save_hyperparameters({"config": config}, logger=False)
        self.model = model
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    def compute_metrics(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self(batch)
        # cross_entropy applies the softmax itself, so the model returns raw scores.
        loss = F.cross_entropy(logits, batch["label"])
        accuracy = (logits.argmax(dim=1) == batch["label"]).float().mean()
        return loss, accuracy

    def _log_metrics(self, stage: str, batch: dict, loss: torch.Tensor, accuracy: torch.Tensor):
        size = batch["label"].shape[0]
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, batch_size=size)
        self.log(f"{stage}/acc", accuracy, on_step=False, on_epoch=True, batch_size=size)

    def training_step(self, batch: dict, batch_index: int) -> torch.Tensor:
        loss, accuracy = self.compute_metrics(batch)
        self._log_metrics("train", batch, loss, accuracy)
        return loss

    def validation_step(self, batch: dict, batch_index: int):
        self._log_metrics("val", batch, *self.compute_metrics(batch))

    def test_step(self, batch: dict, batch_index: int):
        self._log_metrics("test", batch, *self.compute_metrics(batch))

    def on_before_optimizer_step(self, optimizer):
        # Measured before any clipping, to decide whether clipping is needed.
        self.log("train/grad_norm", grad_norm(self, norm_type=2)["grad_2.0_norm_total"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)

    def on_save_checkpoint(self, checkpoint: dict):
        # The embedding table never changes, so it's left out of every checkpoint.
        state = checkpoint["state_dict"]
        for key in [key for key in state if "embedding" in key]:
            del state[key]

    def on_load_checkpoint(self, checkpoint: dict):
        checkpoint["state_dict"].update({k: v for k, v in self.state_dict().items() if "embedding" in k})


class BestValidation(L.Callback):
    """Records the epoch with the lowest validation loss, for sorting runs by it."""

    def __init__(self):
        self.loss = math.inf
        self.accuracy = None
        self.epoch = None

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        loss = trainer.callback_metrics["val/loss"].item()
        if loss < self.loss:
            self.loss = loss
            self.accuracy = trainer.callback_metrics["val/acc"].item()
            self.epoch = trainer.current_epoch
            if isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.summary.update(
                    {"best/val_loss": self.loss, "best/val_acc": self.accuracy, "best/epoch": self.epoch}
                )
