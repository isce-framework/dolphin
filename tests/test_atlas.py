from pathlib import Path

import pytest
import yamale


def test_import():
    import atlas  # noqa: F401


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

    data_min = yamale.make_data(Path(__file__).parent / "data/s1_disp_minimal.yaml")
    yamale.validate(schema, data_min)
    data_full = yamale.make_data(Path(__file__).parent / "data/s1_disp_full.yaml")
    yamale.validate(schema, data_full)
