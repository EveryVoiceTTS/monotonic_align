from monotonic_align import VERSION
from importlib.metadata import version

def test_version_in_sync():
    assert VERSION == version("ilt-monotonic-align")
