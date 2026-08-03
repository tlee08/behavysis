"""Torch models for behavioural classification."""

from .architectures import CNN1, CNN2, DNN1, DNN2, DNN3
from .base import TorchModel

__all__ = [
    "CNN1",
    "CNN2",
    "DNN1",
    "DNN2",
    "DNN3",
    "TorchModel",
]
