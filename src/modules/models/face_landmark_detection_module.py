from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric


class SynergyNetModule(LightningModule):
    """
    LightningModule for SynergyNet

    A `LightningModule` implements 7 key methods:

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
            net: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile:bool
    ):
        """
        Initialize a "SynergyNetModule"
        Args:
            net: The model to train
            optimizer: The optimizer to use for training
            scheduler: The learning rate scheduler to use for training
            compile:
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

        # loss function
        self.criterion = criterion

        # TODO: metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # TODO: for tracking best so far validation accuracy, could be MinMetric that we need
        self.val_metric_best = MaxMetric()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model 'self.net'
        Args:
            x: A tensor of image

        Returns:
            A tensor of landmark?

        """
        return self.net(x)

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        self.val_loss.reset()
        self.val_acc.reset()  # TODO switch these to corresponding metric
        self.val_acc_best.reset()

    def model_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single model step on a batch of data
        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target tensor
        Returns:
            A tuple containing (in order):
                - A tensor of losses.
                - A tensor of predictions.
                - A tensor of target labels.
        """
        x, y = batch
        pred = self.forward(x)
        loss =self.criterion(pred, y)
        # TODO: originaly here is pred = torch.argmax(preds, dim=1), make sure we are right here
        return loss, pred, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target labels.
            batch_idx: The index of the current batch.
        Returns:
             A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets) # TODO: make sure we update our metric
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

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
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets) # TODO: make sure to match this.
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        # TODO: make sure to update this function
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Args:
            stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
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

