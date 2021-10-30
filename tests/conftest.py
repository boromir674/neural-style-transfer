import pytest


@pytest.fixture
def test_suite():
    """Path of the test suite directory."""
    import os
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def test_image(test_suite):
    import os
    def get_image_file_path(file_name):
        return os.path.join(test_suite, 'data', file_name)
    return get_image_file_path


@pytest.fixture
def disk():
    from neural_style_transfer.disk_operations import Disk
    return Disk
