import click

from .disk_operations import Disk
from .styling_observer import StylingObserver
from .algorithm import NSTAlgorithm, AlogirthmParameters
from .nst_tf_algorithm import NSTAlgorithmRunner
from .termination_condition.termination_condition import TerminationConditionFacility
from .termination_condition_adapter import TerminationConditionAdapterFactory
from .nst_image import ImageManager


@click.command()
@click.argument('content_image')
@click.argument('style_image')
@click.option('--interactive', '-i', type=bool, default=True)
@click.option('--iterations', '-it', type=int, default=100)
@click.option('--location', '-l', type=str, default='.')
def cli(content_image, style_image, interactive, iterations, location):

    TERMINATION_CONDITION = 'max-iterations'
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2),
    ]

    import numpy as np
    from artificial_artwork.image.image_operations import noisy, reshape_image, subtract, convert_to_uint8

    # todo dynamically find means
    means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))  # means

    image_manager = ImageManager([
        lambda matrix: reshape_image(matrix, ((1,) + matrix.shape)),
        lambda matrix: subtract(matrix, means),  # input image must have 3 channels!
    ])

    # probably load each image in separate thread and then join
    image_manager.load_from_disk(content_image, 'content')
    image_manager.load_from_disk(style_image, 'style')

    if not image_manager.images_compatible:
        print("Given CONTENT image '{content_image}' has 'height' x 'width' x "
        f"'color_channels': {image_manager.content_image.matrix.shape}")
        print("Given STYLE image '{style_image}' has 'height' x 'width' x "
        f"'color_channels': {image_manager.style_image.matrix.shape}")
        print('Expected to receive images (matrices) of identical shape')
        print('Exiting..')
        exit(1)

    # image_factory = ImageFactory(Disk.load_image)
    # content_image = image_factory.from_disk(content_image, reshape_and_normalize_pipeline)
    # style_image = image_factory.from_disk(style_image, reshape_and_normalize_pipeline)   

    termination_condition = TerminationConditionFacility.create(
        TERMINATION_CONDITION, iterations)
    termination_condition_adapter = TerminationConditionAdapterFactory.create(
        TERMINATION_CONDITION, termination_condition)
    print(f' -- Termination Condition: {termination_condition}')

    algorithm_parameters = AlogirthmParameters(
        image_manager.content_image,
        image_manager.style_image,
        STYLE_LAYERS,
        termination_condition_adapter,
        location,
    )

    algorithm = NSTAlgorithm(algorithm_parameters)

    # algorithm_runner = NSTAlgorithmRunner.default(
    #     algorithm,
    #     image_factory.image_processor.noisy
    # )

    noisy_ratio = 0.6  # ratio

    # NEW
    algorithm_runner = NSTAlgorithmRunner.default(
        algorithm,
        lambda matrix: noisy(matrix, noisy_ratio)
    )

    algorithm_runner.progress_subject.add(
        termination_condition_adapter,
    )
    algorithm_runner.peristance_subject.add(
        StylingObserver(Disk.save_image, convert_to_uint8)
    )

    algorithm_runner.run()
