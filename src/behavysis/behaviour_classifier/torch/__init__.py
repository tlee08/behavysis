"""Torch models for behavioural classification."""

from .architectures import CNN, DNN1, DNN2, DNN3, MODEL_TYPES
from .base import TorchModel

__all__ = [
    "CNN",
    "DNN1",
    "DNN2",
    "DNN3",
    "MODEL_TYPES",
    "TorchModel",
]
