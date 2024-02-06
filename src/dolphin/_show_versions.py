"""Utility methods to print system info for debugging.

Adapted from `rasterio.show_versions`,
which was adapted from `sklearn.utils._show_versions`
which was adapted from `pandas.show_versions`
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Optional

import dolphin

__all__ = ["show_versions"]


def _get_sys_info() -> dict[str, str]:
    """System information.

    Returns
    -------
    dict
        system and Python version information

    """
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "machine": platform.platform(),
    }


def _get_version(module_name: str) -> Optional[str]:
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            return None
    try:
        return mod.__version__
    except AttributeError:
        return mod.version


def _get_opera_info() -> dict[str, Optional[str]]:
    """Information on isce/opera specific modules.

    Returns
    -------
    dict
        dolphin / opera module information

    """
    import opera_utils

    return {
        "dolphin": dolphin.__version__,
        "opera_utils": opera_utils.__version__,
        # optionals
        "isce3": _get_version("isce3"),
        "tophu": _get_version("tophu"),
    }


def _get_deps_info() -> dict[str, Optional[str]]:
    """Overview of the installed version of main dependencies.

    Returns
    -------
    dict:
        version information on relevant Python libraries

    """
    deps = [
        "numpy",
        "numba",
        "jax",
        "osgeo.gdal",
        "h5py",
        "ruamel_yaml",
        "pydantic",
        "setuptools",
    ]
    return {name: _get_version(name) for name in deps}


def _get_gpu_info() -> dict[str, Optional[str]]:
    """Overview of the optional GPU packages.

    Returns
    -------
    dict:
        version information on relevant Python libraries

    """
    from dolphin.utils import gpu_is_available

    return {"jax": _get_version("jax"), "gpu_is_available": str(gpu_is_available())}


def _print_info_dict(info_dict: dict) -> None:
    """Print the information dictionary."""
    for key, stat in info_dict.items():
        print(f"{key:>12}: {stat}")


def show_versions() -> None:
    """Print useful debugging information.

    Examples
    --------
    > python -c "import dolphin; dolphin.show_versions()"

    """
    from dolphin.utils import gpu_is_available

    print("dolphin/isce info:")
    _print_info_dict(_get_opera_info())
    print("\nSystem:")
    _print_info_dict(_get_sys_info())
    print("\nPython deps:")
    _print_info_dict(_get_deps_info())
    print("optional GPU info:")
    print(f"{gpu_is_available() = }")
    _print_info_dict(_get_gpu_info())
