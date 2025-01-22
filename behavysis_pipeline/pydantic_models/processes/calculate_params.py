from behavysis_pipeline.pydantic_models.pydantic_base_model import PydanticBaseModel


class CheckExistsConfigs(PydanticBaseModel):
    bodyparts: list[str] | str = []
    window_sec: float | str = 0
    pcutoff: float | str = 0


class StartFrameFromCsvConfigs(PydanticBaseModel):
    csv_fp: str = ""
    name: str | None = None


class StopFrameConfigs(PydanticBaseModel):
    dur_sec: float | str = 0


class PxPerMmConfigs(PydanticBaseModel):
    pt_a: str = "pt_a"
    pt_b: str = "pt_b"
    pcutoff: int | str = 0
    dist_mm: float | str = 0


class CalculateParamsConfigs(PydanticBaseModel):
    start_frame: CheckExistsConfigs = CheckExistsConfigs()
    start_frame_from_csv: StartFrameFromCsvConfigs = StartFrameFromCsvConfigs()
    stop_frame: StopFrameConfigs = StopFrameConfigs()
    exp_dur: CheckExistsConfigs = CheckExistsConfigs()
    px_per_mm: PxPerMmConfigs = PxPerMmConfigs()
