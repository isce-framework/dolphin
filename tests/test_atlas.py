from pathlib import Path

import pytest
import yamale


def test_import():
    import atlas  # noqa: F401


@pytest.fixture
def schema_file():
    from atlas.utils import get_yaml_file

    return get_yaml_file("s1_disp.yaml", yaml_type="schemas")


@pytest.fixture
def defaults_file():
    from atlas.utils import get_yaml_file

    return get_yaml_file("s1_disp.yaml", yaml_type="defaults")


@pytest.fixture
def data_file():
    return Path(__file__).parent / "data/s1_disp_test.yaml"


def test_yaml_loading(schema_file, defaults_file, data_file):
    # print('!!'*20)
    schema = yamale.make_schema(schema_file)
    defaults = yamale.make_data(defaults_file)
    # print(schema)
    # print('!!'*20)
    # print(d)
    # print('!!'*20)

    # The defaults should fail the schema
    with pytest.raises(yamale.yamale_error.YamaleError):
        yamale.validate(schema, defaults)

    data = yamale.make_data(data_file)
    yamale.validate(schema, data)
