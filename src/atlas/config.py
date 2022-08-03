import datetime
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Union

import yamale
from ruamel.yaml import YAML

Pathlike = Union[PathLike[str], str]


def load_workflow_yaml(input_path: Pathlike, *, workflow_name: str = "s1_disp"):
    """Load and validate a yaml file for a workflow.

    Parameters
    ----------
    input_path : Pathlike
        Path to the yaml file to load
    workflow_name : str
        Name of the workflow to load. Used to determine the path to the
        schema and defaults files.

    Returns
    -------
    data : dict
        Dictionary containing the yaml data
    """
    parser = YAML(typ="safe")
    with open(input_path, "r") as f:
        supplied = parser.load(f)

    defaults_path = get_workflow_yaml_path(name=workflow_name, yaml_type="defaults")
    with open(defaults_path, "r") as f:
        defaults = parser.load(f)

    updated = deep_update(defaults=defaults, supplied=supplied)
    # d = yamale.make_data([(updated, None)])

    schema_path = get_workflow_yaml_path(name=workflow_name, yaml_type="schemas")
    schema = yamale.make_schema(schema_path)
    yamale.validate(schema, [(updated, None)])
    return updated


def save_yaml(output_path: Pathlike, data: dict):
    """Save a yaml file for a workflow.

    Used to record the default-filled version of a supplied yaml.

    Parameters
    ----------
    data : dict
        Dictionary containing the yaml data
    output_path : Pathlike
        Path to the yaml file to save
    """
    parser = YAML(typ="safe")
    with open(output_path, "w") as f:
        parser.dump(data, f)


def add_atlas_section(cfg):
    """Add package and runtime metadata to a loaded config.

    Parameters
    ----------
    cfg : dict
        Loaded configuration dict from `load_yaml`

    Returns
    -------
    cfg : dict
        Configuration dict with added "atlas" section
    """
    from atlas import __version__

    atlas_cfg = {
        "version": __version__,
        "runtime": str(datetime.datetime.now()),
        # TODO: anything else relevant?
    }
    cfg["atlas"] = atlas_cfg
    return cfg


def get_workflow_yaml_path(name: str = "s1_disp.yaml", yaml_type: str = "schemas"):
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
    outpath = Path(__file__).parent / yaml_type / name
    if outpath.suffix != ".yaml":
        outpath = outpath.with_suffix(".yaml")
    return outpath


def deep_update(*, defaults: dict, supplied: dict, copy: bool = True):
    """Update the defaults of a dict with user-supplied dict.

    Parameters
    ----------
    defaults : dict
        Dict with default options to be updated
    supplied: dict
        Dict with user-defined options used to supplied defaults/default
    copy: bool
        Whether to copy the defaults dict before updating

    Returns
    -------
    updated: dict
        default dictionary, updated with user-defined options.
        If copy is True, this is a deep copy of defaults.


    References
    ----------
    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    updated = deepcopy(defaults) if copy else defaults
    for key, val in supplied.items():
        if isinstance(val, dict):
            updated[key] = deep_update(defaults=updated.get(key, {}), supplied=val)
        else:
            updated[key] = val

    return updated
