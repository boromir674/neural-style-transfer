
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



@click.command()
@click.argument('content_image')
@click.argument('style_image')
@click.option('--interactive', '-i', type=bool, default=True)
@click.option('--location', '-l', type=str, default='nst_output')
def cli(content_image, style_image, interactive, location):

    # my_dir = os.path.dirname(os.path.realpath(__file__))
    IMAGE_MODEL_PATH = os.path.join('/', 'data', 'repos', 'neural-style-transfer', 'imagenet-vgg-verydeep-19.mat')
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
    
    termination_condition = TerminationConditionFacility.create(TERMINATION_CONDITION, 100)
    termination_condition_adapter = TerminationConditionAdapterFactory.create(TERMINATION_CONDITION, termination_condition)

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
