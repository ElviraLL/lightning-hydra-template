"""
Training Endpoint, train the model according to the configuration
Author: Jingwen Liang
Email: jingwen@genies.com
Version: v1
Date: Sep 7, 2023
"""
from typing import Any, Dict, List, Optional, Tuple

import pyrootutils
import lightning as L
import hydra
from omegaconf import DictConfig
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger

from src import utils

# --------------------------------------------------------------------------- #
# `pyrootutils.setup_root(...)` below is optional line to make environment more
# convenient should be placed at the top of each entry file
#
# main advantages:
# - allows you to keep all entry files in "src/" without installing project as
#   a package
# - launching python file works no matter where is your current work dir
# - automatically loads environment variables from ".env" if exists
#
# how it works:
# - `setup_root()` above recursively searches for either ".git" or
#   "pyproject.toml" in present and parent dirs, to determine the project root
#   dir
# - adds root dir to the PYTHONPATH (if `pythonpath=True`), so this file can
#   be run from any place without installing project as a package
# - sets PROJECT_ROOT environment variable which is used in
#   "configs/paths/default.yaml" to make all paths always relative to project
#   root
# - loads environment variables from ".env" in root dir (if `dotenv=True`)
#
# you can remove `pyrootutils.setup_root(...)` if you:
# 1. either install project as a package or move each entry file to the project
#    root dir
# 2. remove PROJECT_ROOT variable from paths in "configs/paths/default.yaml"
#
# https://github.com/ashleve/pyrootutils
# --------------------------------------------------------------------------- #
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".project-root"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "train.yaml",
}
log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a test set, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, which applies
    extra utilities before and after the call. It controls the behavior during
    failure. Useful for multi-runs, saving info about the crash, etc.

    Args:
        cfg: A DictConfig configuration composed by Hydra.

    Returns:
         Tuple[dict, dict]: A tuple with metrics and dict with all instantiated objects.
    """
    utils.log_gpu_memory_metadata()

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info("Seed everything with <%d>", cfg.seed)
        L.seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    log.info("Instantiating datamodule <%s>", cfg.data._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Init lightning model
    log.info("Instantiating model <%s>", cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Init callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    # Init loggers
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # Init lightning ddp plugins
    log.info("Instantiating plugins...")
    plugins: Optional[List[Any]] = utils.instantiate_plugins(cfg)

    log.info("Instantiating trainer <%s>", cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, plugins=plugins)

    # Send parameters from cfg to all lightning loggers
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Log metadata
    log.info("Logging metadata!")
    utils.log_metadata(cfg)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info("Best ckpt path: %s", ckpt_path)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> Optional[float]:
    """
    Main entry point of training
    Args:
        cfg: DictConfig configuration composed by Hydra
    """

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()
