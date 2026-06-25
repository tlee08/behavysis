from pathlib import Path

from pydantic import BaseModel

from behavysis.constants import BPTS_SIMBA


class FromLikelihoodConfig(BaseModel):
    """FromLikelihoodConfig."""

    bodyparts: list[str] | str = BPTS_SIMBA
    window_sec: float | str = 1.0
    pcutoff: float | str = 0.8


class StartFrameFromCsvConfig(BaseModel):
    """StartFrameFromCsvConfig."""

    csv_fp: Path = Path("path_to") / "start_times.csv"
    name: str | None = None


class StopFrameFromDurConfig(BaseModel):
    """StopFrameFromDurConfig."""

    dur_sec: float | str = 6000.0


class PxPerMmConfig(BaseModel):
    """PxPerMmConfig."""

    pt_a: str = "pt_a"
    pt_b: str = "pt_b"
    pcutoff: float | str = 0.5
    dist_mm: float | str = 400.0


class CalculateParamsConfig(BaseModel):
    """CalculateParamsConfig."""

    from_likelihood: FromLikelihoodConfig = FromLikelihoodConfig()
    start_frame_from_csv: StartFrameFromCsvConfig = StartFrameFromCsvConfig()
    stop_frame_from_dur: StopFrameFromDurConfig = StopFrameFromDurConfig()
    px_per_mm: PxPerMmConfig = PxPerMmConfig()
