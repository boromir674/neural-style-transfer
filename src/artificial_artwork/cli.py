import sys
import click
import numpy as np

from .disk_operations import Disk
from .styling_observer import StylingObserver
from .algorithm import NSTAlgorithm, AlogirthmParameters
from .nst_tf_algorithm import NSTAlgorithmRunner
from .termination_condition.termination_condition import TerminationConditionFacility
from .termination_condition_adapter import TerminationConditionAdapterFactory
from .nst_image import ImageManager
from .image.image_operations import noisy, reshape_image, subtract, convert_to_uint8
from .production_networks import NetworkDesign
from .pretrained_model.model_handler import ModelHandlerFacility


def load_pretrained_model_functions():
    from .pre_trained_models import vgg
    return vgg


def read_images(content, style):
    # todo dynamically find means
    means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))  # means

    image_manager = ImageManager([
        lambda matrix: reshape_image(matrix, ((1,) + matrix.shape)),
        lambda matrix: subtract(matrix, means),  # input image must have 3 channels!
    ])

    # probably load each image in separate thread and then join
    image_manager.load_from_disk(content, 'content')
    image_manager.load_from_disk(style, 'style')

    if not image_manager.images_compatible:
        print("Given CONTENT image '{content_image}' has 'height' x 'width' x "
        f"'color_channels': {image_manager.content_image.matrix.shape}")
        print("Given STYLE image '{style_image}' has 'height' x 'width' x "
        f"'color_channels': {image_manager.style_image.matrix.shape}")
        print('Expected to receive images (matrices) of identical shape')
        print('Exiting..')
        sys.exit(1)
    
    return image_manager.content_image, image_manager.style_image


@click.command()
@click.argument('content_image')
@click.argument('style_image')
@click.option('--iterations', '-it', type=int, default=100, show_default=True)
@click.option('--location', '-l', type=str, default='.')
def cli(content_image, style_image, iterations, location):

    termination_condition = 'max-iterations'

    content_image, style_image = read_images(content_image, style_image)

    load_pretrained_model_functions()
    model_design = type('ModelDesign', (), {
        'pretrained_model': ModelHandlerFacility.create('vgg'),
        'network_design': NetworkDesign.from_default_vgg()
    })
    model_design.pretrained_model.load_model_layers()

    termination_condition_instance = TerminationConditionFacility.create(
        termination_condition, iterations)
    termination_condition_adapter = TerminationConditionAdapterFactory.create(
        termination_condition, termination_condition_instance)
    print(f' -- Termination Condition: {termination_condition_instance}')

    algorithm_parameters = AlogirthmParameters(
        content_image,
        style_image,
        termination_condition_adapter,
        location,
    )

    algorithm = NSTAlgorithm(algorithm_parameters)

    noisy_ratio = 0.6  # ratio

    algorithm_runner = NSTAlgorithmRunner.default(
        lambda matrix: noisy(matrix, noisy_ratio),
    )

    algorithm_runner.progress_subject.add(
        termination_condition_adapter,
    )
    algorithm_runner.persistance_subject.add(
        StylingObserver(Disk.save_image, convert_to_uint8)
    )

    algorithm_runner.run(algorithm, model_design)
