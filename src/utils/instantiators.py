from typing import Any, Callable, List, Optional

import hydra
from omegaconf import DictConfig

from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger


from src.utils import pylogger, rich_utils

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg (DictConfig): Callbacks config.

    Returns:
        List[Callback]: List with all instantiated callbacks.
    """

    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info("Instantiating callback <%s>", cb_conf._target_)
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_cfg (DictConfig): Loggers config.

    Returns:
        List[LightningLoggerBase]: List with all instantiated loggers.
    """

    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <%s>", lg_conf._target_)
            logger.append(hydra.utils.instantiate(lg_conf))
    return logger


def instantiate_plugins(cfg: DictConfig) -> Optional[List[Any]]:
    """Instantiates lightning plugins from config.

    Args:
        cfg (DictConfig): Config.

    Returns:
        List[Any]: List with all instantiated plugins.
    """

    if not cfg.extras.get("plugins"):
        log.warning("No plugins configs found! Skipping...")
        return

    if cfg.trainer.get("accelerator") == "cpu":
        log.warning("Using CPU as accelerator! Skipping...")
        return

    plugins: List[Any] = []
    for _, pl_conf in cfg.extras.get("plugins").items():
        if isinstance(pl_conf, DictConfig) and "_target_" in pl_conf:
            log.info(f"Instantiating plugin <{pl_conf._target_}>")
            plugins.append(hydra.utils.instantiate(pl_conf))

    return plugins
