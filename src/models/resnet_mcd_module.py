from typing import Any, Dict, Tuple

import numpy as np
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.resnet import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from src.utils.uncertainty import (
    AleatoricUncertainty,
    EpistemicUncertainty,
    TotalUncertainty,
)


class ResnetMCDLitModule(LightningModule):
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
        mcd_samples_train: int,
        mcd_samples_val: int,
        mcd_samples_test: int,
        optimizer: torch.optim.Optimizer,
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
        elif net_config["arch"] == "resnet50":
            self.net = ResNet50(
                in_channels=net_config["in_channels"],
                num_classes=net_config["num_classes"],
                dropout_rate=self.dropout_rate,
            )
        elif net_config["arch"] == "resnet101":
            self.net = ResNet101(
                in_channels=net_config["in_channels"],
                num_classes=net_config["num_classes"],
                dropout_rate=self.dropout_rate,
            )
        elif net_config["arch"] == "resnet152":
            self.net = ResNet152(
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

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], mcd_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step with Monte Carlo Dropout on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        :param mcd_samples: The number of stochastic MC samples (forward passed)
        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of stochastic predictions [TxBxC].
            - A tensor of predictions [BxC].
            - A tensor of target labels [B].
        """
        x, y = batch
        mcd_preds = []
        self.net.train()  # ensure dropout is active
        for _ in range(mcd_samples):
            logits = self.forward(x)
            mcd_preds.append(logits)

        mcd_preds = torch.stack(mcd_preds)  # [T, B, C] to compute uncertainties
        mean_preds = mcd_preds.mean(dim=0)  # [B, C] to compute accuracy

        loss = self.criterion(mean_preds, y)
        preds = torch.argmax(mean_preds, dim=1)

        return loss, mcd_preds, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step with Monte Carlo Dropout on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, mcd_preds, preds, targets = self.model_step(
            batch, mcd_samples=self.hparams.mcd_samples_train
        )

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_tu(mcd_preds, probs=False)
        self.train_au(mcd_preds, probs=False)
        self.train_eu(mcd_preds, probs=False)
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
        loss, mcd_preds, preds, targets = self.model_step(
            batch, mcd_samples=self.hparams.mcd_samples_val
        )

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_tu(mcd_preds, probs=False)
        self.val_au(mcd_preds, probs=False)
        self.val_eu(mcd_preds, probs=False)
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
        loss, mcd_preds, preds, targets = self.model_step(
            batch, mcd_samples=self.hparams.mcd_samples_test
        )

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_tu(mcd_preds, probs=False)
        self.test_au(mcd_preds, probs=False)
        self.test_eu(mcd_preds, probs=False)
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
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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
    _ = ResnetMCDLitModule(None, None, None, None)
