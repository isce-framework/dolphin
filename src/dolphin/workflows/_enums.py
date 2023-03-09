from enum import Enum

__all__ = [
    "WorkflowName",
    "OutputFormat",
    "ShpMethod",
    "UnwrapMethod",
    "InterferogramNetworkType",
]


class WorkflowName(str, Enum):
    """Name of workflows."""

    STACK = "stack"
    SINGLE = "single"


class OutputFormat(str, Enum):
    """Raster format for the final workflow output."""

    ENVI = "ENVI"
    GTIFF = "GTiff"
    NETCDF = "NetCDF"


class ShpMethod(str, Enum):
    """Method for finding SHPs during phase linking."""

    KL = "KL"
    KS = "KS"
    RECT = "rect"
    # Alias for no SHP search
    NONE = "rect"


class UnwrapMethod(str, Enum):
    """Phase unwrapping method, passable to Tophu."""

    SNAPHU = "snaphu"
    ICU = "icu"


class InterferogramNetworkType(str, Enum):
    """Type of interferogram network to create from phase-linking results."""

    SINGLE_REFERENCE = "single-reference"
    MANUAL_INDEX = "manual-index"
    MAX_BANDWIDTH = "max-bandwidth"
    MAX_TEMPORAL_BASELINE = "max-temporal-baseline"
