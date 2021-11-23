import pytest

from click.testing import CliRunner
from artificial_artwork.cli import cli
from unittest.mock import patch


runner = CliRunner()

@pytest.fixture
def image_file_names():
    return type('Images', (), {
        'content': 'canoe_water_w300-h225.jpg',
        'style': 'blue-red_w300-h225.jpg'
    })


@pytest.fixture
def image_manager(image_manager_class):
    """Production ImageManager instance."""
    import numpy as np
    from artificial_artwork.image.image_operations import reshape_image, subtract
    means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    return image_manager_class([
        lambda matrix: reshape_image(matrix, ((1,) + matrix.shape)),
        lambda matrix: subtract(matrix, means),  # input image must have 3 channels!
    ])


@pytest.fixture
def max_iterations_adapter_factory_method():
    from artificial_artwork.termination_condition.termination_condition import TerminationConditionFacility
    from artificial_artwork.termination_condition_adapter import TerminationConditionAdapterFactory

    def create_max_iterations_termination_condition_adapter(iterations):
        return TerminationConditionAdapterFactory.create(
            'max-iterations', TerminationConditionFacility.create(
                'max-iterations', iterations
        ))
    return create_max_iterations_termination_condition_adapter


@pytest.fixture
def algorithm_parameters_class():
    from artificial_artwork.algorithm import AlogirthmParameters
    return AlogirthmParameters


@pytest.fixture
def algorithm(algorithm_parameters_class):
    from artificial_artwork.algorithm import NSTAlgorithm
    def _create_algorithm(*parameters):
        return NSTAlgorithm(algorithm_parameters_class(*parameters))
    return _create_algorithm


@pytest.fixture
def create_algorithm(algorithm, pre_trained_model, tmpdir):
    def _create_algorithm(image_manager, termination_condition_adapter):
        return algorithm(
            image_manager.content_image,
            image_manager.style_image,
            pre_trained_model.style_layers,
            termination_condition_adapter,
            tmpdir
        )
    return _create_algorithm


@pytest.fixture
def create_production_algorithm_runner():
    from artificial_artwork.nst_tf_algorithm import NSTAlgorithmRunner
    from artificial_artwork.image.image_operations import noisy, convert_to_uint8
    from artificial_artwork.styling_observer import StylingObserver
    from artificial_artwork.disk_operations import Disk
    
    noisy_ratio = 0.6
    def _create_production_algorithm_runner(termination_condition_adapter, model_design):
        algorithm_runner = NSTAlgorithmRunner.default(
            lambda matrix: noisy(matrix, noisy_ratio),
            model_design
        )

        algorithm_runner.progress_subject.add(
            termination_condition_adapter,
        )
        algorithm_runner.persistance_subject.add(
            StylingObserver(Disk.save_image, convert_to_uint8)
        )
        return algorithm_runner
    return _create_production_algorithm_runner


@pytest.fixture
def get_algorithm_runner(create_production_algorithm_runner, pre_trained_model):
    def _get_algorithm_runner(termination_condition_adapter):
        algorithm_runner = create_production_algorithm_runner(
            termination_condition_adapter,
            type('ModelDesign', (), {
                'network_layers': pre_trained_model.network_layers,
                'parameters_loader': pre_trained_model.parameters_loader
            })
        )
        algorithm_runner.NETWORK_OUTPUT = pre_trained_model.output_layer
        return algorithm_runner
    return _get_algorithm_runner


def test_nst_runner(
    get_algorithm_runner,
    create_algorithm,
    image_file_names,
    max_iterations_adapter_factory_method,
    image_manager,
    test_image,
    pre_trained_model,
    tmpdir):
    """Test nst algorithm runner.

    Runs a simple 'smoke test' by iterating only 3 times.
    """
    import os
    ITERATIONS = 3

    image_manager.load_from_disk(test_image(image_file_names.content), 'content')
    image_manager.load_from_disk(test_image(image_file_names.style), 'style')

    assert image_manager.images_compatible == True

    termination_condition_adapter = max_iterations_adapter_factory_method(ITERATIONS)

    algorithm_runner = get_algorithm_runner(termination_condition_adapter)

    algorithm = create_algorithm(image_manager, termination_condition_adapter)

    algorithm_runner.run(algorithm)

    template_string = image_file_names.content + '+' + image_file_names.style + '-' + '{}' + '.png'
    assert os.path.isfile(os.path.join(tmpdir, template_string.format(1)))
    assert os.path.isfile(os.path.join(tmpdir, template_string.format(ITERATIONS)))
