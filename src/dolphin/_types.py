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

    Attributes
    ----------
        ERA5 (str): ERA5 Tropospheric Model.
        HRES (str): HRES Tropospheric Model.
        ERAINT (str): ERAINT Tropospheric Model.
        ERAI (str): ERAI Tropospheric Model.
        MERRA (str): MERRA Tropospheric Model.
        NARR (str): NARR Tropospheric Model.
        HRRR (str): HRRR Tropospheric Model.
        GMAO (str): GMAO Tropospheric Model.
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
    """Enumeration representing different tropospheric types.

    Attributes
    ----------
        WET (str): Wet Tropospheric Type.
        DRY (str): Dry Tropospheric Type.
        HYDROSTATIC (str): Hydrostatic Tropospheric Type.
        COMB (str): Combined Tropospheric Type.
    """

    WET = "wet"
    DRY = "dry"
    HYDROSTATIC = "hydrostatic"
    COMB = "comb"
