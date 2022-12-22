"""Utility methods to print system info for debugging.

Adapted from `rasterio.show_versions`,
which was adapted from `sklearn.utils._show_versions`
which was adapted from `pandas.show_versions`
"""
import importlib
import platform
import sys
import typing

import dolphin

__all__ = ["show_versions"]


def _get_sys_info() -> typing.Dict[str, str]:
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


def _get_opera_info():
    """Information on system on core modules.

    Returns
    -------
    dict
        dolphin / opera module information
    """
    blob = {
        "dolphin": dolphin.__version__,
        # optionals
        "isce3": _get_version("isce3"),
        "compass": _get_version("compass"),
    }
    return blob


def _get_version(modname):
    if modname in sys.modules:
        mod = sys.modules[modname]
    else:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            return None
    try:
        return mod.__version__
    except AttributeError:
        return mod.version


def _get_deps_info():
    """Overview of the installed version of main dependencies.

    Returns
    -------
    dict:
        version information on relevant Python libraries
    """
    deps = [
        "numpy",
        "numba",
        "osgeo.gdal",
        "h5py",
        "ruamel_yaml",
        "pydantic",
        "setuptools",
    ]
    return {name: _get_version(name) for name in deps}


def _print_info_dict(info_dict):
    """Print the information dictionary."""
    for key, stat in info_dict.items():
        print(f"{key:>12}: {stat}")


def show_versions():
    """Print useful debugging information.

    Examples
    --------
    > python -c "import dolphin; dolphin.show_versions()"
    """
    print("dolphin info:")
    _print_info_dict(_get_opera_info())
    print("\nSystem:")
    _print_info_dict(_get_sys_info())
    print("\nPython deps:")
    _print_info_dict(_get_deps_info())
