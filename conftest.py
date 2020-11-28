# content of conftest.py
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runtrain", action="store_true", default=False, help="run train mode tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "train: mark test as using older transformers for training")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--runtrain"):
        skip_slow = pytest.mark.skip(reason="need --runtrain option to run")
        for item in items:
            if "train" in item.keywords:
                item.add_marker(skip_slow)
