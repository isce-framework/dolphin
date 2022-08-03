import datetime
from pathlib import Path

import pytest
import yamale

import atlas.config
from atlas.config import get_workflow_yaml_path


# Using fixtures for these to avoid nagivating the paths from tests/ to src/
@pytest.fixture
def schema_file():
    return get_workflow_yaml_path("s1_disp.yaml", yaml_type="schemas")


@pytest.fixture
def defaults_file():
    return get_workflow_yaml_path("s1_disp.yaml", yaml_type="defaults")


def test_yaml_loading(schema_file, defaults_file):
    schema = yamale.make_schema(schema_file)
    defaults = yamale.make_data(defaults_file)

    # The defaults should fail the schema
    with pytest.raises(yamale.yamale_error.YamaleError):
        yamale.validate(schema, defaults)

    # Check that the updating of defaults works
    minimal_path = Path(__file__).parent / "data/s1_disp_minimal.yaml"
    min_data = atlas.config.load_workflow_yaml(minimal_path, workflow_name="s1_disp")
    assert min_data["nmap"]["pvalue"] == 0.05

    default_dict = yamale.make_data(defaults_file)[0][0]
    for k, v in default_dict.items():
        if k != "input_vrt_file":
            assert min_data[k] == v


def test_yaml_save(tmp_path):
    from ruamel.yaml import YAML

    minimal_path = Path(__file__).parent / "data/s1_disp_minimal.yaml"
    min_data = atlas.config.load_workflow_yaml(minimal_path, workflow_name="s1_disp")

    temp_file = tmp_path / "out.yaml"
    atlas.config.save_yaml(temp_file, min_data)
    y = YAML()
    assert y.load(temp_file) == min_data


def test_atlas_cfg_section():
    minimal_path = Path(__file__).parent / "data/s1_disp_minimal.yaml"
    min_data = atlas.config.load_workflow_yaml(minimal_path, workflow_name="s1_disp")
    cfg_augmented = atlas.config.add_atlas_section(min_data)

    now = str(datetime.datetime.now())
    # '2022-08-03 08:47:28.834660' , ignore microseconds
    assert cfg_augmented["atlas"]["runtime"][:-7] == now[:-7]
