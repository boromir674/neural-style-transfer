import pytest


@pytest.fixture
def image_manager():
    from artificial_artwork.nst_image import ImageManager
    return ImageManager([lambda array: array + 2])


@pytest.fixture
def compatible_images(test_image):
    return type('CompatibleImages', (), {
        'content': test_image('canoe_water.jpg'),
        'style': test_image('blue-red-w400-h300.jpg'),
    })()

@pytest.fixture
def incompatible_image(test_image):
    return test_image('wikipedia-logo.png')


def test_image_manager(image_manager, compatible_images, incompatible_image):
    assert image_manager.images_compatible == False

    image_manager.load_from_disk(compatible_images.content, 'content')
    assert image_manager.images_compatible == False

    image_manager.load_from_disk(compatible_images.style, 'style')
    assert image_manager.images_compatible == True

    image_manager.load_from_disk(incompatible_image, 'content')
    assert image_manager.images_compatible == False

    with pytest.raises(ValueError):
        image_manager.load_from_disk(compatible_images.content, 'unknown-type')
