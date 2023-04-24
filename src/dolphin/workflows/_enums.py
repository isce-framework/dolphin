from enum import Enum

__all__ = [
    "WorkflowName",
    "ShpMethod",
    "UnwrapMethod",
    "InterferogramNetworkType",
]


class WorkflowName(str, Enum):
    """Name of workflows."""

    STACK = "stack"
    SINGLE = "single"


class ShpMethod(str, Enum):
    """Method for finding SHPs during phase linking."""

    TF = "tf"
    KL = "kl"
    KS = "ks"
    RECT = "rect"
    # Alias for no SHP search
    NONE = "rect"


class UnwrapMethod(str, Enum):
    """Phase unwrapping method."""

    SNAPHU = "snaphu"
    ICU = "icu"


class InterferogramNetworkType(str, Enum):
    """Type of interferogram network to create from phase-linking results."""

    SINGLE_REFERENCE = "single-reference"
    MANUAL_INDEX = "manual-index"
    MAX_BANDWIDTH = "max-bandwidth"
    MAX_TEMPORAL_BASELINE = "max-temporal-baseline"
