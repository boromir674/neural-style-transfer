import pytest


@pytest.fixture
def toy_image_manager(image_manager_class):
    return image_manager_class([lambda array: array + 2])


@pytest.fixture
def compatible_images(test_image):
    return type(
        "CompatibleImages",
        (),
        {
            "content": test_image("canoe_water.jpg"),
            "style": test_image("blue-red-w400-h300.jpg"),
        },
    )()


@pytest.fixture
def incompatible_image(test_image):
    return test_image("wikipedia-logo.png")


def test_image_manager(toy_image_manager, compatible_images, incompatible_image):
    assert toy_image_manager.images_compatible == False

    toy_image_manager.load_from_disk(compatible_images.content, "content")
    assert toy_image_manager.images_compatible == False

    toy_image_manager.load_from_disk(compatible_images.style, "style")
    assert toy_image_manager.images_compatible == True

    toy_image_manager.load_from_disk(incompatible_image, "content")
    assert toy_image_manager.images_compatible == False

    with pytest.raises(ValueError):
        toy_image_manager.load_from_disk(compatible_images.content, "unknown-type")
