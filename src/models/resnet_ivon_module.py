from typing import Any, Dict, Tuple

import ivon
import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.resnet import ResNet18, ResNet34
from src.utils.uncertainty import (
    AleatoricUncertainty,
    EpistemicUncertainty,
    TotalUncertainty,
)


class ResnetIVONLitModule(LightningModule):
    """Example of a `LightningModule` for CIFAR10 classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net_config: Dict,
        train_samples: int,
        test_samples: int,
        lr: float,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.dropout_rate = net_config["dropout_rate"]
        if net_config["arch"] == "resnet18":
            self.net = ResNet18(
                in_channels=net_config["in_channels"],
                num_classes=net_config["num_classes"],
                dropout_rate=self.dropout_rate,
            )
        elif net_config["arch"] == "resnet34":
            self.net = ResNet34(
                in_channels=net_config["in_channels"],
                num_classes=net_config["num_classes"],
                dropout_rate=self.dropout_rate,
            )
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=net_config["num_classes"])
        self.val_acc = Accuracy(task="multiclass", num_classes=net_config["num_classes"])
        self.test_acc = Accuracy(task="multiclass", num_classes=net_config["num_classes"])
        self.train_tu = TotalUncertainty()
        self.train_au = AleatoricUncertainty()
        self.train_eu = EpistemicUncertainty(self.train_tu, self.train_au)
        self.val_tu = TotalUncertainty()
        self.val_au = AleatoricUncertainty()
        self.val_eu = EpistemicUncertainty(self.val_tu, self.val_au)
        self.test_tu = TotalUncertainty()
        self.test_au = AleatoricUncertainty()
        self.test_eu = EpistemicUncertainty(self.test_tu, self.test_au)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step with Monte Carlo Dropout on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of sampled predictions [TxBxC].
            - A tensor of predictions [BxC].
            - A tensor of target labels [B].
        """
        x, y = batch
        opt = self.optimizers()

        # Predict at mean
        if self.hparams.test_samples == 0:
            logits = self.forward(x)  # [B,C]
            loss = self.criterion(logits, y)
            sampled_probs = F.softmax(logits, dim=1).unsqueeze(0)  # [1, B, C]
        # (or) Predict with samples
        else:
            sampled_probs = []
            for _ in range(self.hparams.test_samples):
                with opt.sampled_params():
                    sampled_logit = self.forward(x)
                    sampled_probs.append(F.softmax(sampled_logit, dim=1))
            sampled_probs = torch.stack(sampled_probs)  # [T, B, C]
            mean_preds = torch.mean(sampled_probs, dim=0)  # [B, C]
            loss = -torch.sum(
                torch.log(mean_preds.clamp(min=1e-6)) * F.one_hot(y, 10), dim=1
            ).mean()

        preds = mean_preds.argmax(dim=1)  # [B, C] for accuracy

        return loss, sampled_probs, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step with Monte Carlo Dropout on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        # loss, sampled_probs, preds, targets = self.model_train_step(batch)
        x, y = batch
        opt = self.optimizers().optimizer
        sampled_probs = []

        for _ in range(self.hparams.train_samples):
            with opt.sampled_params(train=True):
                logits = self.forward(x)
                loss = self.criterion(logits, y)
                self.manual_backward(loss)
                sampled_probs.append(F.softmax(logits, dim=1))

        opt.step()
        opt.zero_grad()
        print(f"TRAIN LOSS {loss}")
        sampled_probs = torch.stack(sampled_probs, dim=0)  # [T, B, C]
        mean_preds = sampled_probs.mean(dim=0)  # [B, C]
        preds = mean_preds.argmax(dim=1)  # [B, C]

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, y)
        self.train_tu(sampled_probs, probs=True)
        self.train_au(sampled_probs, probs=True)
        self.train_eu(sampled_probs, probs=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/tu", self.train_tu, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/au", self.train_au, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/eu", self.train_eu, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, sampled_probs, preds, targets = self.model_test_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_tu(sampled_probs, probs=True)
        self.val_au(sampled_probs, probs=True)
        self.val_eu(sampled_probs, probs=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tu", self.val_tu, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/au", self.val_au, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/eu", self.val_eu, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, sampled_probs, preds, targets = self.model_test_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_tu(sampled_probs, probs=True)
        self.test_au(sampled_probs, probs=True)
        self.test_eu(sampled_probs, probs=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tu", self.test_tu, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/au", self.test_au, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/eu", self.test_eu, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Use IVON custom optimizer

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = ivon.IVON(
            self.trainer.model.parameters(),
            lr=self.hparams.lr,
            beta1=0.9,
            hess_init=0.01,
            ess=25_000,
        )

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ResnetIVONLitModule(None, None, None, None)
