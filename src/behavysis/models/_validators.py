class ConfigNotConfiguredError(ValueError):
    """A config field has not been configured yet."""

    def __init__(self, field: str) -> None:
        """Initialize ConfigNotConfiguredError."""
        super().__init__(
            f"Config field '{field}' is not set",
        )


class MetadataNotReadyError(ValueError):
    """A metadata field has not been computed yet."""

    def __init__(self, field: str, stage: str) -> None:
        """Initialize MetadataNotReadyError."""
        super().__init__(
            f"Metadata field '{field}' is not set. Run '{stage}' first.",
        )
