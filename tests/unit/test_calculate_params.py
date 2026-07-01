"""Unit tests for calculate_params module.

NOTE: These tests require proper keypoints data with the correct column structure.
They are marked as integration tests because they need realistic data.
"""

from pathlib import Path

import pytest
from loguru import logger

from behavysis.funcs import stop_frame_from_dur
from behavysis.models import ExperimentConfig
