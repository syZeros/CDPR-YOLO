"""DDP-compatible bounded runs preserving the original full training schedule."""

import os

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER


def stop_after_prefix(trainer):
    limit = int(os.environ["CDPR_PREFIX_EPOCHS"])
    if trainer.epoch + 1 >= limit:
        LOGGER.info(f"Prefix comparison: stopping after {limit} epochs; schedule remains {trainer.epochs} epochs.")
        trainer.stop = True


class PrefixDetectionTrainer(DetectionTrainer):
    """Re-register the stop callback in each spawned DDP process."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if int(os.environ.get("CDPR_PREFIX_EPOCHS", "0")) < 1:
            raise ValueError("CDPR_PREFIX_EPOCHS must be positive for PrefixDetectionTrainer")
        self.add_callback("on_fit_epoch_end", stop_after_prefix)
