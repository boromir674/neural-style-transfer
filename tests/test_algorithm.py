import pytest

# from click.testing import CliRunner


# runner = CliRunner()


@pytest.fixture
def image_file_names():
    return type(
        "Images",
        (),
        {"content": "canoe_water_w300-h225.jpg", "style": "blue-red_w300-h225.jpg"},
    )


@pytest.fixture
def image_manager(image_manager_class):
    """Production ImageManager instance."""
    import numpy as np

    from artificial_artwork.image.image_operations import reshape_image, subtract

    means = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return image_manager_class(
        [
            lambda matrix: reshape_image(matrix, ((1,) + matrix.shape)),
            lambda matrix: subtract(matrix, means),  # input image must have 3 channels!
        ]
    )


@pytest.fixture
def max_iterations_adapter_factory_method():
    from artificial_artwork.termination_condition_adapter_factory import (
        TerminationConditionAdapterFactory,
    )

    def create_max_iterations_termination_condition_adapter(iterations):
        return TerminationConditionAdapterFactory.create("max-iterations", iterations)

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
def create_algorithm(algorithm, tmpdir):
    def _create_algorithm(image_manager, termination_condition_adapter):
        return algorithm(
            image_manager.content_image,
            image_manager.style_image,
            termination_condition_adapter,
            tmpdir,
        )

    return _create_algorithm


@pytest.fixture
def create_production_algorithm_runner():
    from artificial_artwork.disk_operations import Disk
    from artificial_artwork.image.image_operations import convert_to_uint8, noisy
    from artificial_artwork.nst_tf_algorithm import NSTAlgorithmRunner
    from artificial_artwork.styling_observer import StylingObserver

    noisy_ratio = 0.6

    def _create_production_algorithm_runner(termination_condition_adapter, max_iterations):
        algorithm_runner = NSTAlgorithmRunner.default(
            lambda matrix: noisy(matrix, noisy_ratio),
        )

        algorithm_runner.progress_subject.add(
            termination_condition_adapter,
        )
        algorithm_runner.persistance_subject.add(
            StylingObserver(Disk.save_image, convert_to_uint8, max_iterations)
        )
        return algorithm_runner

    return _create_production_algorithm_runner


@pytest.fixture
def get_algorithm_runner(create_production_algorithm_runner):
    def _get_algorithm_runner(termination_condition_adapter, max_iterations):
        algorithm_runner = create_production_algorithm_runner(
            termination_condition_adapter,
            max_iterations,
        )
        return algorithm_runner

    return _get_algorithm_runner


@pytest.fixture
def get_model_design():
    def _get_model_design(handler, network_design):
        return type(
            "ModelDesign", (), {"pretrained_model": handler, "network_design": network_design}
        )

    return _get_model_design


def test_nst_runner(
    get_algorithm_runner,
    create_algorithm,
    image_file_names,
    get_model_design,
    max_iterations_adapter_factory_method,
    image_manager,
    test_image,
    model,
    tmpdir,
):
    """Test nst algorithm runner.

    Runs a simple 'smoke test' by iterating only 3 times.
    """
    import os

    ITERATIONS = 3

    image_manager.load_from_disk(test_image(image_file_names.content), "content")
    image_manager.load_from_disk(test_image(image_file_names.style), "style")

    assert image_manager.images_compatible == True

    termination_condition_adapter = max_iterations_adapter_factory_method(ITERATIONS)

    algorithm_runner = get_algorithm_runner(termination_condition_adapter, ITERATIONS)

    algorithm = create_algorithm(image_manager, termination_condition_adapter)

    model_design = get_model_design(
        model.pretrained_model.handler,
        model.network_design,
    )
    model_design.pretrained_model.load_model_layers()
    algorithm_runner.run(algorithm, model_design)

    template_string = (
        image_file_names.content + "+" + image_file_names.style + "-" + "{}" + ".png"
    )
    assert os.path.isfile(os.path.join(tmpdir, template_string.format(1)))
    assert os.path.isfile(os.path.join(tmpdir, template_string.format(ITERATIONS)))
