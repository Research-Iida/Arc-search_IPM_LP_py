import pytest

from src.infra.julia.setup_julia import setup_julia


@pytest.fixture(scope="function")
def fixture_setup_julia():
    """julia の activate は1セッションで1回実行すれば十分"""
    setup_julia()
    yield
