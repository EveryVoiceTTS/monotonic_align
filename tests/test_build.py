from importlib.metadata import version

from monotonic_align import VERSION


def test_version_in_sync():
    assert VERSION == version("ilt-monotonic-align")


if __name__ == "__main__":
    test_version_in_sync()
    print("OK")
