import pytest
import numpy as np


@pytest.fixture
def image_operations():
    from artificial_artwork.image.image_operations import reshape_image, subtract, noisy, convert_to_uint8
    return type('Ops', (), {
        'reshape': reshape_image,
        'subtract': subtract,
        'noisy': noisy,
        'convert_to_uint8': convert_to_uint8,
    })


@pytest.mark.parametrize('test_image', [
    ([[11,12,13], [21, 22, 23]]),
])
def test_image_reshape(test_image, image_operations):
    math_array = np.array(test_image, dtype=np.float32)
    image = image_operations.reshape(math_array, (1,) + math_array.shape)
    assert image.shape == (1,) + math_array.shape


@pytest.mark.parametrize('test_image, array', [
    ([[11,12,13], [21, 22, 23]], [[1, 2, 3], [4, 5, 6]]),
])
def test_subtract_image(test_image, array, image_operations):
    math_array_1 = np.array(test_image, dtype=np.float32)
    math_array_2 = np.array(array, dtype=np.float32)

    image = image_operations.subtract(math_array_1, math_array_2)
    assert image.shape == math_array_1.shape
    assert image.tolist() == [
        [10, 10, 10],
        [17, 17, 17]
    ]

@pytest.mark.parametrize('test_image, array', [
    ([[11,12,13], [21, 22, 23]], [[1, 2], [4, 5]]),
])
def test_wrong_subtract(test_image, array, image_operations):
    from artificial_artwork.image.image_operations import ShapeMissmatchError
    with pytest.raises(ShapeMissmatchError):
        math_array_1 = np.array(test_image, dtype=np.float32)
        math_array_2 = np.array(array, dtype=np.float32)
        image_operations.subtract(math_array_1, math_array_2)



@pytest.mark.parametrize('test_image, ratio', [
    ([[11,12,13], [21, 22, 23]], 0),
    ([[11,12,13], [21, 22, 23]], 0.6),
    ([[11,12,13], [21, 22, 23]], 1),
])
def test_noisy(test_image, ratio, image_operations):
    math_array = np.array(test_image, dtype=np.float32)
    min_pixel_value = np.min(math_array)
    max_pixel_value = np.max(math_array)
    image = image_operations.noisy(math_array, ratio)
    assert (image <= max(max_pixel_value, 20)).all()
    assert (min(min_pixel_value, -20) <= image).all()



@pytest.mark.parametrize('test_image, ratio', [
    ([[11, 12], [21, 22]], 1.1),
    ([[12, 13], [21, 23]], -0.2),
])
def test_wrong_noisy_ratio(test_image, ratio, image_operations):
    from artificial_artwork.image.image_operations import InvalidRatioError
    math_array = np.array(test_image, dtype=np.float32)
    with pytest.raises(InvalidRatioError):
        image = image_operations.noisy(math_array, ratio)


# UINT8 CONVERTION TESTS
@pytest.mark.parametrize('test_image, expected_image', [
(
        [[1.2, 9.1],
        [10, 3]],

        [[0, 229],
        [255, 52]]
    ),

    (
        [[1, 1],
        [3, 5]],

        [[0, 0],
        [127, 255]]
    ),

    (
        [[1, 1],
        [1, 1]],

        [[1, 1],
        [1, 1]]
    ),

])
def test_uint8_convertion(test_image, expected_image, image_operations):
    runtime_image = image_operations.convert_to_uint8(np.array(test_image, dtype=np.float32))
    assert runtime_image.dtype == np.uint8
    assert 0 <= np.nanmin(runtime_image)
    assert np.nanmax(runtime_image) < np.power(2.0, 8)
    assert runtime_image.tolist() == expected_image


@pytest.mark.parametrize('test_image', [
    (
        [[np.nan, np.nan],
        [np.nan, np.nan]],
    ),

    (
        [[1, -float('inf')],
        [2, 3]],
    ),

])
def test_non_finite_minimum_value(test_image, image_operations):
    with pytest.raises(ValueError, match=r'Minimum image value is not finite'):
        runtime_image = image_operations.convert_to_uint8(np.array(test_image, dtype=np.float32))


@pytest.mark.parametrize('test_image', [
    (
        [[1, float('inf')],
        [2, 3]],
    ),
])
def test_non_finite_maximum_value(test_image, image_operations):
    with pytest.raises(ValueError, match=r'Maximum image value is not finite'):
        runtime_image = image_operations.convert_to_uint8(np.array(test_image, dtype=np.float32))
