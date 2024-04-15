from enum import Enum

__all__ = [
    "ShpMethod",
    "UnwrapMethod",
    "CallFunc",
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


class CallFunc(str, Enum):
    """Call function for the timeseries method to find reference point."""

    MIN = "min"
    MAX = "max"
