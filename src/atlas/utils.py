import errno
import os
from pathlib import Path


def mkdir_p(path):
    """Emulates bash `mkdir -p`, in python style."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_yaml_file(name="s1_disp.yaml", yaml_type="schemas"):
    """Get the path to a yaml schema or default file.

    Parameters
    ----------
    name : str
        Name of the schema
    yaml_type : str, choices = ["schemas", "defaults"]
        Which type of yaml file to get

    Returns
    -------
    path : str
        Path to the schema
    """
    if yaml_type not in ["schemas", "defaults"]:
        raise ValueError("yaml_type must be one of ['schemas', 'defaults']")
    return Path(__file__).parent / yaml_type / name


def deep_update(original, supplied):
    """Update the defaults of a dict with user-supplied dict.

    Parameters
    ----------
    original : dict
        Dict with default options to be updated
    supplied: dict
        Dict with user-defined options used to supplied original/default

    Returns
    -------
    original: dict
        Default dictionary updated with user-defined options

    References
    ----------
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for key, val in supplied.items():
        if isinstance(val, dict):
            original[key] = deep_update(original.get(key, {}), val)
        else:
            original[key] = val

    # return updated original
    return original
