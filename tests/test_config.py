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
    min_data = atlas.config.load_yaml(minimal_path, workflow_name="s1_disp")
    assert min_data["nmap"]["pvalue"] == 0.05
