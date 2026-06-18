"""Pytest configuration and shared fixtures for behavysis tests."""

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Path to the test project fixture data
TEST_PROJECT_DIR = Path(__file__).parent / "data"


# =============================================================================
# Session-scoped fixtures (created once per test session)
# =============================================================================


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "test"


@pytest.fixture(scope="session")
def sample_project_dir(test_data_dir: Path) -> Path:
    """Return path to sample project directory with test videos."""
    return test_data_dir / "project"


@pytest.fixture(scope="session")
def sample_video_path(sample_project_dir: Path) -> Path | None:
    """Return path to a sample video file, or None if not found."""
    raw_vid_dir = sample_project_dir / "1_raw_vid"
    if raw_vid_dir.exists():
        videos = list(raw_vid_dir.glob("*.mp4"))
        if videos:
            return videos[0]
    return None


# =============================================================================
# DataFrame fixtures for unit tests
# =============================================================================


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Return a simple DataFrame for basic tests."""
    return pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
        index=pd.Index([0, 1, 2], name="frame"),
    )


@pytest.fixture
def multiindex_df() -> pd.DataFrame:
    """Return a DataFrame with MultiIndex for testing DFMixin."""
    index = pd.MultiIndex.from_tuples(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        names=["frame", "subframe"],
    )
    columns = pd.MultiIndex.from_tuples(
        [("x", "nose"), ("y", "nose"), ("x", "tail"), ("y", "tail")],
        names=["coord", "bodypart"],
    )
    data = np.random.randn(4, 4)
    return pd.DataFrame(data, index=index, columns=columns)


@pytest.fixture
def keypoints_df_data() -> pd.DataFrame:
    """Return synthetic keypoints data matching KeypointsDf schema.

    Structure:
        Index: (frame,) - frame numbers
        Columns: (scorer, individuals, bodyparts, coords) - multi-level
    """
    n_frames = 100
    bodyparts = [
        "Nose",
        "BodyCentre",
        "TailBase1",
        "LeftEar",
        "RightEar",
        "LeftFlankMid",
        "RightFlankMid",
    ]
    individuals = ["mouse1marked", "mouse2unmarked"]
    coords = ["x", "y", "likelihood"]
    scorer = [" scorer"]

    # Create multi-index columns
    columns = pd.MultiIndex.from_product(
        [scorer, individuals, bodyparts, coords],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )

    # Create frame index
    index = pd.Index(range(n_frames), name="frame")

    # Generate synthetic data
    np.random.seed(42)
    data = np.random.randn(n_frames, len(columns))

    # Make x, y coordinates reasonable (centred around frame center)
    n_cols_per_indiv = len(bodyparts) * len(coords)
    for i in range(len(individuals)):
        for j in range(len(bodyparts)):
            col_offset = i * n_cols_per_indiv + j * len(coords)
            # x coordinates (0-1000 range roughly)
            data[:, col_offset] = np.random.uniform(100, 900, n_frames)
            # y coordinates (0-500 range roughly)
            data[:, col_offset + 1] = np.random.uniform(50, 450, n_frames)
            # likelihood (0-1 range)
            data[:, col_offset + 2] = np.random.uniform(0.5, 1.0, n_frames)

    return pd.DataFrame(data, index=index, columns=columns)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide a temporary directory for file operations."""
    yield tmp_path
    # Cleanup happens automatically via tmp_path fixture


@pytest.fixture
def temp_parquet_file(temp_dir: Path) -> Path:
    """Provide a temporary parquet file path."""
    return temp_dir / "test_data.parquet"


# =============================================================================
# Config fixtures
# =============================================================================


@pytest.fixture
def minimal_config_dict() -> dict:
    """Return minimal valid config dictionary."""
    return {
        "user": {
            "format_vid": {
                "height_px": 540,
                "width_px": 960,
                "fps": 15,
            },
        },
        "ref": {
            "bodyparts-centre": ["BodyCentre", "TailBase1"],
        },
    }
