"""Constants package for behavysis."""

from behavysis.constants.bodypoints import (
    BPTS_CENTRE,
    BPTS_CORNERS,
    BPTS_FRONT,
    BPTS_SIMBA,
    INDIVS_SIMBA,
    INDIVS_SINGLE,
)
from behavysis.constants.gui import (
    CHECKSTATE2VALUE,
    COLOR2VALUE,
    QIMAGE_FORMAT,
    STATUS_MSG_TIMEOUT,
    VALUE2CHECKSTATE,
    VALUE2COLOR,
)
from behavysis.constants.pipeline import (
    ANALYSIS_DIR,
    CACHE_DIR,
    DF_IO_FORMAT,
    DIAGNOSTICS_DIR,
    FileExts,
    Folders,
)
from behavysis.constants.plot import PLOT_DPI, PLOT_STYLE, STR_DIV

__all__ = [
    "ANALYSIS_DIR",
    "BPTS_CENTRE",
    "BPTS_CORNERS",
    "BPTS_FRONT",
    "BPTS_SIMBA",
    "CACHE_DIR",
    "CHECKSTATE2VALUE",
    "COLOR2VALUE",
    "DF_IO_FORMAT",
    "DIAGNOSTICS_DIR",
    "INDIVS_SIMBA",
    "INDIVS_SINGLE",
    "PLOT_DPI",
    "PLOT_STYLE",
    "QIMAGE_FORMAT",
    "STATUS_MSG_TIMEOUT",
    "STR_DIV",
    "VALUE2CHECKSTATE",
    "VALUE2COLOR",
    "FileExts",
    "Folders",
]
