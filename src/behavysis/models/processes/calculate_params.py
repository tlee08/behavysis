import os

from pydantic import BaseModel

from behavysis.constants import BPTS_SIMBA


class FromLikelihoodConfigs(BaseModel):
    bodyparts: list[str] | str = BPTS_SIMBA
    window_sec: float | str = 1.0
    pcutoff: float | str = 0.8


class StartFrameFromCsvConfigs(BaseModel):
    csv_fp: str = os.path.join("path_to", "start_times.csv")
    name: str | None = None


class StopFrameFromDurConfigs(BaseModel):
    dur_sec: float | str = 6000


class PxPerMmConfigs(BaseModel):
    pt_a: str = "pt_a"
    pt_b: str = "pt_b"
    pcutoff: float | str = 0.5
    dist_mm: float | str = 400


class CalculateParamsConfigs(BaseModel):
    from_likelihood: FromLikelihoodConfigs = FromLikelihoodConfigs()
    start_frame_from_csv: StartFrameFromCsvConfigs = StartFrameFromCsvConfigs()
    stop_frame_from_dur: StopFrameFromDurConfigs = StopFrameFromDurConfigs()
    px_per_mm: PxPerMmConfigs = PxPerMmConfigs()
