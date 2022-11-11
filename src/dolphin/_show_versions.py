"""Utility methods to print system info for debugging.

adapted from :func:`rasterio.show_versions`,
which was adapted from :func:`sklearn.utils._show_versions`
which was adapted from :func:`pandas.show_versions`
"""
import importlib
import platform
import sys

from rich import print


def _get_sys_info():
    """System information.

    Returns
    -------
    dict:
        system and Python version information
    """
    blob = [
        ("python", sys.version.replace("\n", " ")),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_gdal_info():
    """Information on system GDAL.

    Returns
    -------
    dict:
        system GDAL information
    """
    # pylint: disable=import-outside-toplevel
    import dolphin

    blob = [
        ("dolphin", dolphin.__version__),
    ]

    return dict(blob)


def _get_deps_info():
    """Overview of the installed version of main dependencies.

    Returns
    -------
    dict:
        version information on relevant Python libraries
    """
    deps = [
        "osgeo.gdal",
        "numpy",
        "ruamel_yaml",
        "yamale",
        "h5py",
        "setuptools",
    ]

    def get_version(module):
        try:
            return module.__version__
        except AttributeError:
            return module.version

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            deps_info[modname] = get_version(mod)
        except ImportError:
            deps_info[modname] = None

    return deps_info


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
    _print_info_dict(_get_gdal_info())
    print("\nSystem:")
    _print_info_dict(_get_sys_info())
    print("\nPython deps:")
    _print_info_dict(_get_deps_info())
