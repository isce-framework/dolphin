import pytest


# https://github.com/pytest-dev/pytest/issues/1872
# Used to create a temporary directory for the tests,
# in which all the tests run.
# This way, the tests don't pollute the working directory
# with the directories they create
@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture(scope="session", autouse=True)
def my_temp_dir(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp("my_temp_dir")
    yield temp_dir
    # Clean up the temporary directory after the tests are finished
    temp_dir.remove()


@pytest.fixture(autouse=True, scope="session")
def change_session_test_dir(my_temp_dir, monkeypatch_session):
    monkeypatch_session.chdir(my_temp_dir)
