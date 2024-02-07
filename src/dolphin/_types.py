from __future__ import annotations

import datetime
import sys
from enum import Enum
from os import PathLike
from typing import TYPE_CHECKING, NamedTuple, TypeVar, Union

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

# Some classes are declared as generic in stubs, but not at runtime.
# In Python 3.9 and earlier, os.PathLike is not subscriptable, results in runtime error
if TYPE_CHECKING:
    from builtins import ellipsis

    Index = ellipsis | slice | int
    PathLikeStr = PathLike[str]
else:
    PathLikeStr = PathLike


PathOrStr = Union[str, PathLikeStr]
Filename = PathOrStr  # May add a deprecation notice for `Filename`
# TypeVar added for generic functions which should return the same type as the input
PathLikeT = TypeVar("PathLikeT", str, PathLikeStr)


class Bbox(NamedTuple):
    """Bounding box named tuple, defining extent in cartesian coordinates.

    Usage:

        Bbox(left, bottom, right, top)

    Attributes
    ----------
    left : float
        Left coordinate (xmin)
    bottom : float
        Bottom coordinate (ymin)
    right : float
        Right coordinate (xmax)
    top : float
        Top coordinate (ymax)

    """

    left: float
    bottom: float
    right: float
    top: float


class Strides(NamedTuple):
    """Decimation/striding factor in the y (column) and x (row) directions."""

    y: int
    x: int


class HalfWindow(NamedTuple):
    """Half-window size in the y (column) and x (row) directions."""

    y: int
    x: int


# Used for callable types
T = TypeVar("T")
P = ParamSpec("P")

DateOrDatetime = Union[datetime.datetime, datetime.date]


class TropoModel(str, Enum):
    """Enumeration representing different tropospheric models."""

    ERA5 = "ERA5"
    HRES = "HRES"
    ERAINT = "ERAINT"
    ERAI = "ERAI"
    MERRA = "MERRA"
    NARR = "NARR"
    HRRR = "HRRR"
    GMAO = "GMAO"


class TropoType(str, Enum):
    """Type of tropospheric delay."""

    WET = "wet"
    """Wet tropospheric delay."""
    DRY = "dry"
    """Dry delay (same as hydrostatic, named "dry" in PyAPS)"""
    HYDROSTATIC = "hydrostatic"
    """Hydrostatic (same as dry, named differently in raider)"""
    COMB = "comb"
    """Combined wet + dry delay."""
