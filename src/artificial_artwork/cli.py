import os
import sys
import click

from .disk_operations import Disk
from .styling_observer import StylingObserver
from .algorithm import NSTAlgorithm, AlogirthmParameters
from .nst_tf_algorithm import NSTAlgorithmRunner
from .termination_condition_adapter_factory import TerminationConditionAdapterFactory
from .nst_image import noisy, convert_to_uint8
from .production_networks import NetworkDesign
from .pretrained_model import ModelHandlerFacility
from .utils import load_pretrained_model_functions, read_images

from artificial_artwork import __version__

this_file_location = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))


# CLI --version flag
def version_msg():
    """artificial_artwork CLI version, lib location and Python version.

    Get message about artificial_artwork version, location
    and Python version.
    """
    # extract everything about version: major, minor, patch and build notes
    python_version = sys.version
    message = u"Neural Style Transfer CLI %(version)s from {} (Python {})"
    location = os.path.dirname(this_file_location)
    return message.format(location, python_version)


# MAIN
@click.group()
@click.version_option(__version__, u'-V', u'--version', message=version_msg())
def entry_point():
    pass


# RUN CMD
@click.command()
@click.argument('content_image')
@click.argument('style_image')
@click.option('--iterations', '-it', type=int, default=100, show_default=True)
@click.option('--location', '-l', type=str, default='.')
def run(content_image, style_image, iterations, location):

    termination_condition = 'max-iterations'

    content_image, style_image = read_images(content_image, style_image)

    load_pretrained_model_functions()
    model_design = type('ModelDesign', (), {
        'pretrained_model': ModelHandlerFacility.create('vgg'),
        'network_design': NetworkDesign.from_default_vgg()
    })
    model_design.pretrained_model.load_model_layers()

    termination_condition = TerminationConditionAdapterFactory.create(
        termination_condition,
        iterations,
    )

    print(f' -- Termination Condition: {termination_condition.termination_condition}')

    algorithm_parameters = AlogirthmParameters(
        content_image,
        style_image,
        termination_condition,
        location,
    )

    algorithm = NSTAlgorithm(algorithm_parameters)

    noisy_ratio = 0.6  # ratio

    algorithm_runner = NSTAlgorithmRunner.default(
        lambda matrix: noisy(matrix, noisy_ratio),
    )

    algorithm_runner.progress_subject.add(
        termination_condition,
    )
    algorithm_runner.persistance_subject.add(
        StylingObserver(Disk.save_image, convert_to_uint8)
    )

    algorithm_runner.run(algorithm, model_design)


### NST CLI Entrypoint ###

# ATTACH CMDs
entry_point.add_command(run)
from .cmd_demo import demo
entry_point.add_command(demo)
