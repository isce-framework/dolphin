from enum import Enum

__all__ = [
    "WorkflowName",
    "OutputFormat",
    "UnwrapMethod",
    "InterferogramNetworkType",
]


class WorkflowName(str, Enum):
    """Names of workflows."""

    STACK = "stack"
    SINGLE = "single"


class OutputFormat(str, Enum):
    """Possible output formats for the workflow."""

    ENVI = "ENVI"
    GTIFF = "GTiff"
    NETCDF = "NetCDF"


class UnwrapMethod(str, Enum):
    """Methods passable to Tophu unwrapping functions."""

    SNAPHU = "snaphu"
    ICU = "icu"


class InterferogramNetworkType(str, Enum):
    """Types of interferogram networks."""

    SINGLE_REFERENCE = "single-reference"
    MAX_BANDWIDTH = "max-bandwidth"
    MAX_TEMPORAL_BASELINE = "max-temporal-baseline"
