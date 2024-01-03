from __future__ import annotations

import datetime
import sys
from enum import Enum
from os import PathLike
from typing import TYPE_CHECKING, Tuple, TypeVar, Union

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

# Some classes are declared as generic in stubs, but not at runtime.
# In Python 3.9 and earlier, os.PathLike is not subscriptable, results in a runtime error
# https://stackoverflow.com/questions/71077499/typeerror-abcmeta-object-is-not-subscriptable
if TYPE_CHECKING:
    PathLikeStr = PathLike[str]
else:
    PathLikeStr = PathLike


PathOrStr = Union[str, PathLikeStr]
Filename = PathOrStr  # May add a deprecation notice for `Filename`
# TypeVar added for generic functions which should return the same type as the input
PathLikeT = TypeVar("PathLikeT", str, PathLikeStr)

# left, bottom, right, top
Bbox = Tuple[float, float, float, float]

# Used for callable types
T = TypeVar("T")
P = ParamSpec("P")

DateOrDatetime = Union[datetime.datetime, datetime.date]


class TropoModel(Enum):
    """Enumeration representing different tropospheric models.
    """

    ERA5 = "ERA5"
    HRES = "HRES"
    ERAINT = "ERAINT"
    ERAI = "ERAI"
    MERRA = "MERRA"
    NARR = "NARR"
    HRRR = "HRRR"
    GMAO = "GMAO"


class TropoType(Enum):
    """Type of tropospheric delay."""

    WET = "wet"
    """Wet tropospheric delay."""
    DRY = "dry"
    """Dry delay (same as hydrostatic, named "dry" in PyAPS)"""
    HYDROSTATIC = "hydrostatic"
    """Hydrostatic (same as dry, named differently in raider)"""
    COMB = "comb"
    """Combined wet + dry delay."""
