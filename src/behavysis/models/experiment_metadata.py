"""Experiment configuration models for the behavysis pipeline."""

from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt

# ═══════════════════════════════════════════════════════════════════════════════
# Config Not Configured Error
# ═══════════════════════════════════════════════════════════════════════════════


class MetadataNotReadyError(ValueError):
    """A metadata field has not been computed yet."""

    def __init__(self, field: str, stage: str) -> None:
        """Initialize MetadataNotReadyError."""
        super().__init__(
            f"Metadata field '{field}' is not set. Run '{stage}' first.",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata Configs
# ═══════════════════════════════════════════════════════════════════════════════


class VideoMetadata(BaseModel):
    """VidMetadata."""

    width_px: PositiveInt | None = None
    height_px: PositiveInt | None = None
    fps: PositiveFloat | None = None
    total_frames: PositiveInt | None = None


class ExperimentMetadata(BaseModel):
    """Experiment Metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    px_per_mm: PositiveFloat | None = None
    start_frame: PositiveInt | None = None
    stop_frame: PositiveInt | None = None
    dur_frames: PositiveInt | None = None
    raw_video: VideoMetadata = VideoMetadata()
    formatted_video: VideoMetadata = VideoMetadata()

    def require_name(self) -> str:
        """Name."""
        if self.name is None:
            msg = "name"
            raise MetadataNotReadyError(
                msg,
                "calculate_parameters.name()",
            )
        return self.name

    def require_px_per_mm(self) -> PositiveFloat:
        """Pixels per MM."""
        if self.px_per_mm is None:
            msg = "px_per_mm"
            raise MetadataNotReadyError(
                msg,
                "calculate_parameters.px_per_mm()",
            )
        return self.px_per_mm

    def require_start_frame(self) -> PositiveInt:
        """Start frame."""
        if self.start_frame is None:
            msg = "start_frame"
            raise MetadataNotReadyError(
                msg,
                "calculate_parameters.start_frame_from_*()",
            )
        return self.start_frame

    def require_stop_frame(self) -> PositiveInt:
        """Stop frame."""
        if self.stop_frame is None:
            msg = "stop_frame"
            raise MetadataNotReadyError(
                msg,
                "calculate_parameters.stop_frame_from_*()",
            )
        return self.stop_frame

    def require_fps(self) -> PositiveFloat:
        """Formatted vid fps."""
        if self.formatted_video.fps is None:
            msg = "formatted_video.fps"
            raise MetadataNotReadyError(
                msg,
                "format_video()",
            )
        return self.formatted_video.fps

    def require_width_px(self) -> PositiveInt:
        """Formatted vid width_px."""
        if self.formatted_video.width_px is None:
            msg = "formatted_video.width_px"
            raise MetadataNotReadyError(
                msg,
                "format_video()",
            )
        return self.formatted_video.width_px

    def require_height_px(self) -> PositiveInt:
        """Formatted vid height_px."""
        if self.formatted_video.height_px is None:
            msg = "formatted_video.height_px"
            raise MetadataNotReadyError(
                msg,
                "format_video()",
            )
        return self.formatted_video.height_px

    def require_total_frames(self) -> PositiveInt:
        """Formatted vid total_frames."""
        if self.formatted_video.total_frames is None:
            msg = "formatted_video.total_frames"
            raise MetadataNotReadyError(
                msg,
                "format_video()",
            )
        return self.formatted_video.total_frames
