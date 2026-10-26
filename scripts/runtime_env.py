"""Validate and record the user-confirmed historical training environment."""

from __future__ import annotations

import importlib.metadata
import platform
import sys

import torch
import torchvision

import ultralytics


EXPECTED = {
    "python": "3.10.14",
    "torch": "2.4.1+cu121",
    "torchvision": "0.19.1+cu121",
    "cuda_runtime": "12.1",
    "cudnn": 90100,
}


def environment_snapshot():
    snapshot = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "framework": ultralytics.__version__,
        "framework_source": ultralytics.__file__,
    }
    snapshot["dependencies"] = {}
    for name in ("numpy", "opencv-python", "pillow", "matplotlib", "pyyaml", "scipy", "polars", "ultralytics-thop"):
        try:
            snapshot["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            snapshot["dependencies"][name] = None
    return snapshot


def validate_environment(snapshot, allow_mismatch=False):
    differences = {key: {"expected": value, "actual": snapshot.get(key)}
                   for key, value in EXPECTED.items() if snapshot.get(key) != value}
    if differences and not allow_mismatch:
        raise RuntimeError(
            f"Training environment differs from the confirmed reproduction environment: {differences}. "
            "Activate the Python 3.10 / PyTorch 2.4.1 CUDA 12.1 environment described in environment.yml. "
            "Use --allow-env-mismatch only for an intentional cross-environment experiment."
        )
    return differences
