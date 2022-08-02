from pathlib import Path

import pytest
import yamale

import atlas.utils


# Using fixtures for these to avoid nagivating the paths from tests/ to src/
@pytest.fixture
def schema_file():
    from atlas.utils import get_yaml_file

    return get_yaml_file("s1_disp.yaml", yaml_type="schemas")


@pytest.fixture
def defaults_file():
    from atlas.utils import get_yaml_file

    return get_yaml_file("s1_disp.yaml", yaml_type="defaults")


def test_yaml_loading(schema_file, defaults_file):
    schema = yamale.make_schema(schema_file)
    defaults = yamale.make_data(defaults_file)

    # The defaults should fail the schema
    with pytest.raises(yamale.yamale_error.YamaleError):
        yamale.validate(schema, defaults)

    # Check that the updating of defaults works
    minimal_path = Path(__file__).parent / "data/s1_disp_minimal.yaml"
    min_data = atlas.utils.load_yaml(minimal_path, workflow_name="s1_disp")
    assert min_data["nmap"]["pvalue"] == 0.05

    # data_min = atlas.utils.load_and_validate_yaml(minimal_path)
    # updated = atlas.utils.deep_update(original=defaults[0][0], supplied=data_min)
    # yamale.validate(schema, updated)
