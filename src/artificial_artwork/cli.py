
import os
import click

from .disk_operations import Disk
from .styling_observer import StylingObserver
from .algorithm import NSTAlgorithm, AlogirthmParameters
from .nst_tf_algorithm import NSTAlgorithmRunner
from .image import ImageFactory, ImageProcessingConfig
from .algorithm_progress import NSTAlgorithmProgress
from .termination_condition.termination_condition import TerminationConditionFacility
from .termination_condition_adapter import TerminationConditionAdapterFactory


def get_vgg_verydeep_19_model():
    try:
        return os.environ['AA_VGG_19']
    except KeyError:
        file_path = os.path.join(os.getcwd(), 'imagenet-vgg-verydeep-19.mat')
        if os.path.exists(file_path):
            return file_path
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'imagenet-vgg-verydeep-19.mat')
        if os.path.exists(file_path):
            return file_path
    raise NoImageModelSpesifiedError('No pretrained image model found. '
        'Please download it and set the AA_VGG_19 environment variable with the'
        'path where ou stored the model (*.mat file), to indicate to wher to '
        'locate and load it')


class NoImageModelSpesifiedError(Exception): pass


@click.command()
@click.argument('content_image')
@click.argument('style_image')
@click.option('--interactive', '-i', type=bool, default=True)
@click.option('--iterations', '-it', type=int, default=100)
@click.option('--location', '-l', type=str, default='.')
def cli(content_image, style_image, interactive, iterations, location):

    IMAGE_MODEL_PATH = get_vgg_verydeep_19_model()
    TERMINATION_CONDITION = 'max-iterations'
    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2),
    ]

    image_factory = ImageFactory(Disk.load_image)

    # for now we have hardcoded the config to receive 300 x 400 images with 3 color channels
    image_process_config = ImageProcessingConfig.from_image_dimensions()
    
    termination_condition = TerminationConditionFacility.create(TERMINATION_CONDITION, iterations)
    termination_condition_adapter = TerminationConditionAdapterFactory.create(TERMINATION_CONDITION, termination_condition)
    print(f' -- Termination Condition: {termination_condition}')

    algorithm_parameters = AlogirthmParameters(
        image_factory.from_disk(content_image),
        image_factory.from_disk(style_image),
        IMAGE_MODEL_PATH,
        STYLE_LAYERS,
        termination_condition_adapter,
        location,
    )

    algorithm = NSTAlgorithm(algorithm_parameters, image_process_config)

    algorithm_runner = NSTAlgorithmRunner.default(algorithm, image_factory.image_processor.noisy)

    algorithm_progress = NSTAlgorithmProgress({})
    styling_observer = StylingObserver(Disk.save_image)
    
    algorithm_runner.progress_subject.add(
        algorithm_progress,
        termination_condition_adapter,
    )
    algorithm_runner.peristance_subject.add(
        styling_observer
    )
            

    algorithm_runner.run()


if __name__ == '__main__':
    cli()
