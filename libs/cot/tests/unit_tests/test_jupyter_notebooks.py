import pytest

from .utils import chdir


def test_jupyter_notebooks():
    """tests if the main cot library can be found and imported in a jupyter notebook"""
    with chdir("notebooks_for_testing"):
        exit_code = pytest.main(["--nbmake", "--tb=no", "import.ipynb"])
    assert int(exit_code) == 0
