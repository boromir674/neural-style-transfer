import pytest


@pytest.mark.parametrize('image_path', [
    ('canoe_water.jpg'),
])
def test_image(image_path, image_factory, test_image):
    in_memory_image = image_factory.from_disk(test_image_path := test_image(image_path), preprocess=True)
    assert in_memory_image.file_path == test_image_path
    assert in_memory_image.matrix.shape == (1, 300, 400, 3)


def test_image_processing_config_methods(default_image_processing_config):
    import numpy as np
    assert default_image_processing_config.image_width == 400
    assert default_image_processing_config.image_height == 300
    assert default_image_processing_config.color_channels == 3
    assert default_image_processing_config.noise_ratio == 0.6
    assert np.array_equal(default_image_processing_config.means, np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)))


@pytest.fixture(params=[
    ['canoe_water.jpg', 400, 300, 4],
    # ['canoe_water.jpg', 300, 300, 3],
], scope='function')
def data(request, test_image):
    import numpy as np
    from neural_style_transfer.image import ImageProcessor, ImageProcessingConfig, ImageFactory, ConfigMeansShapeMissmatchError
    from neural_style_transfer.disk_operations import Disk
    dummy_means = [100 + i * 10 for i in range(request.param[3])]
    return type('WrongConfigurationScenarioData', (), {
        'image_factory': ImageFactory(Disk.load_image, ImageProcessor(ImageProcessingConfig(
            request.param[1],
            request.param[2],
            request.param[3],
            0.6,
            np.array(dummy_means).reshape((1, 1, 1, request.param[3]))
        ))),
        'test_image': test_image(request.param[0]),
        'expected_exception': ConfigMeansShapeMissmatchError,
    })


def test_wrong_configuration(data):
    """Test a failed scenario of image preprocessing.
    
    Tests the behaviour when the input image dimensions and/or color channels do not match the ones configured
    and used by components such as the ImageProcessor. 
    """
    with pytest.raises(data.expected_exception):
        data.image_factory.from_disk(data.test_image, preprocess=True)
