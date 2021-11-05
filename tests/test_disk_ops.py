import pytest


@pytest.fixture
def images_to_save(test_image):
    import imageio as iio
    return {
        'im1': iio.imread(test_image('wikipedia-logo.png'))
    }


@pytest.mark.parametrize('image_id, expected_size', [  # size in Bytes
    ('im1', 198604),
])
def test_save_operation(image_id, expected_size, disk, images_to_save, tmpdir):
    import os
    target_file = os.path.join(tmpdir, image_id)
    disk.save_image(images_to_save[image_id], target_file, format='png')
    # assert actual size of file in disk matches the expected size
    assert os.path.getsize(target_file) == expected_size


@pytest.mark.parametrize('image_name, expected_shape, expected_item_size', [
    ('wikipedia-logo.png', (1058, 1058), 1),
])
def test_load_operation(image_name, expected_shape, expected_item_size, test_image, disk):
    from functools import reduce
    image = disk.load_image(test_image(image_name))
    assert image.shape == expected_shape
    assert image.size == reduce(lambda i, j: i * j, expected_shape)
    assert image.itemsize == expected_item_size  # in Bytes
    assert image.nbytes == expected_item_size * image.size
