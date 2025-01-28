import os

from behavysis.constants import SIMBA_BODYPARTS
from behavysis.pydantic_models.pydantic_base_model import PydanticBaseModel


class FromLikelihoodConfigs(PydanticBaseModel):
    bodyparts: list[str] | str = SIMBA_BODYPARTS
    window_sec: float | str = 1.0
    pcutoff: float | str = 0.8


class StartFrameFromCsvConfigs(PydanticBaseModel):
    csv_fp: str = os.path.join("path_to", "start_times.csv")
    name: str | None = None


class StopFrameFromDurConfigs(PydanticBaseModel):
    dur_sec: float | str = 6000


class PxPerMmConfigs(PydanticBaseModel):
    pt_a: str = "pt_a"
    pt_b: str = "pt_b"
    pcutoff: float | str = 0.8
    dist_mm: float | str = 400


class CalculateParamsConfigs(PydanticBaseModel):
    from_likelihood: FromLikelihoodConfigs = FromLikelihoodConfigs()
    start_frame_from_csv: StartFrameFromCsvConfigs = StartFrameFromCsvConfigs()
    stop_frame_from_dur: StopFrameFromDurConfigs = StopFrameFromDurConfigs()
    px_per_mm: PxPerMmConfigs = PxPerMmConfigs()
