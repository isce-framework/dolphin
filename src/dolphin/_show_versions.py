"""Utility methods to print system info for debugging.

Adapted from `rasterio.show_versions`,
which was adapted from `sklearn.utils._show_versions`
which was adapted from `pandas.show_versions`
"""

from __future__ import annotations

import importlib
import platform
import re
import sys
from importlib.metadata import metadata
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
            mod = importlib.import_module(module_name.replace("-", "_"))
        except ImportError:
            return None
    try:
        return mod.__version__
    except AttributeError:
        return mod.version


def _get_unwrapping_options() -> dict[str, Optional[str]]:
    """Information on possible phase unwrapping libraries.

    Returns
    -------
    dict
        module information

    """
    out = {}
    for unwrapper in ["snaphu", "spurt", "isce3", "tophu", "whirlwind"]:
        out[unwrapper] = _get_version(unwrapper)
        print(f"{unwrapper} : {out[unwrapper]}")
    return out


def _get_deps_info() -> dict[str, Optional[str]]:
    """Overview of the installed version of main dependencies.

    Returns
    -------
    dict:
        version information on relevant Python libraries

    """
    # Get metadata for your package
    meta = metadata("dolphin")
    # Extract dependencies from 'Requires-Dist' field
    deps = [
        re.split(r"[><=~!]", dep.split()[0])[0]
        for dep in meta.get_all("Requires-Dist", [])
        if "extra" not in dep
    ]
    # Replace 'ruamel-yaml' with 'ruamel.yaml'
    deps = [dep.replace("ruamel-yaml", "ruamel.yaml") for dep in deps]
    # Add `osgeo` for gdal (not listed in pip requirements)
    deps += ["osgeo.gdal"]
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
    print(f"dolphin version: {dolphin.__version__}")
    print("\nPython deps:")
    _print_info_dict(_get_deps_info())
    print("\nSystem:")
    _print_info_dict(_get_sys_info())
    print("Unwrapping packages:")
    _print_info_dict(_get_unwrapping_options())
    print("optional GPU info:")
    _print_info_dict(_get_gpu_info())
