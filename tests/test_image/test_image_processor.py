import pytest
import numpy as np


@pytest.fixture
def image_processor():
    from artificial_artwork.image.image_processor import ImageProcessor
    return ImageProcessor()


@pytest.mark.parametrize('image, pipeline, output', [
    ([[1,2,3], [4,5,6]], [], [[1,2,3], [4,5,6]]),
    ([[1,2,3], [4,5,6]], [lambda array: array + 1], [[2,3,4], [5,6,7]]),
])
def test_image_processor(image, pipeline, output, image_processor):
    runtime_output = image_processor.process(np.array(image), pipeline)
    assert (runtime_output - np.array(output) == 0).all()
