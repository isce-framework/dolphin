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
    RECT = "rect"
    # Alias for no SHP search
    NONE = "rect"


class UnwrapMethod(str, Enum):
    """Phase unwrapping method."""

    SNAPHU = "snaphu"
    ICU = "icu"
    PHASS = "phass"
    SPURT = "spurt"
    WHIRLWIND = "whirlwind"


class CallFunc(str, Enum):
    """Call function for the timeseries method to find reference point."""

    MIN = "min"
    MAX = "max"


class CompressedSlcPlan(str, Enum):
    """Plan for creating Compressed SLCs during phase linking."""

    ALWAYS_FIRST = "always_first"
    FIRST_PER_MINISTACK = "first_per_ministack"
    LAST_PER_MINISTACK = "last_per_ministack"
