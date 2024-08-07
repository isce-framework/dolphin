from typing import Any

from .ionosphere import estimate_ionospheric_delay
from .troposphere import estimate_tropospheric_delay

__all__ = [
    "estimate_ionospheric_delay",
    "estimate_tropospheric_delay",
]


def __getattr__(name: str) -> Any:
    if name == "delay_from_netcdf":
        # let's load this module lazily to avoid an ImportError
        # for when we don't use netcdf
        from ._netcdf import delay_from_netcdf

        return delay_from_netcdf

    if name in __all__:
        return globals()[name]

    errmsg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(errmsg)


def __dir__() -> list[str]:
    try:
        from ._netcdf import delay_from_netcdf
    except ModuleNotFoundError:
        return __all__
    else:
        return sorted([*__all__, "delay_from_netcdf"])
