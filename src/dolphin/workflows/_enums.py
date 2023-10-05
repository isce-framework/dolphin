from enum import Enum

__all__ = [
    "ShpMethod",
    "UnwrapMethod",
    "InterferogramNetworkType",
]


class ShpMethod(str, Enum):
    """Method for finding SHPs during phase linking."""

    GLRT = "glrt"
    KS = "ks"
    KLD = "kld"
    RECT = "rect"
    # Alias for no SHP search
    NONE = "rect"


class UnwrapMethod(str, Enum):
    """Phase unwrapping method."""

    SNAPHU = "snaphu"
    ICU = "icu"
    PHASS = "phass"


class InterferogramNetworkType(str, Enum):
    """Type of interferogram network to create from phase-linking results."""

    SINGLE_REFERENCE = "single-reference"
    MANUAL_INDEX = "manual-index"
    MAX_BANDWIDTH = "max-bandwidth"
    MAX_TEMPORAL_BASELINE = "max-temporal-baseline"
