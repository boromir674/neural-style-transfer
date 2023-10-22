from pathlib import Path

from .disk_operations import Disk
from .styling_observer import StylingObserver
from .algorithm import NSTAlgorithm, AlogirthmParameters
from .nst_tf_algorithm import NSTAlgorithmRunner
from .termination_condition_adapter_factory import TerminationConditionAdapterFactory
from .nst_image import noisy, convert_to_uint8
from .production_networks import NetworkDesign
from .pretrained_model import ModelHandlerFacility
from .utils import load_pretrained_model_functions, read_images


# def validate_and_normalize_path(ctx, param, value):
#     """Custom function to validate and normalize a path."""
#     if value is None:
#         return None
#     path = Path(value)

#     if path.is_absolute():
#         abs_path = path
#     else:
#         current_directory = Path.cwd()
#         abs_path = current_directory / path

#     if not abs_path.exists():
#         abs_path.mkdir()
#         click.echo(f'Folder "{abs_path}" created')
#     else:
#         # get files inside the folder
#         folder_files = [f for f in abs_path.iterdir() if f.is_file()]
#         if len(folder_files) > 0:
#             # ask user whether to delete everything, process as it is or exit
#             click.echo(f'Folder "{abs_path}" already exists and is not empty.')
#             click.echo('What do you want to do?')
#             click.echo('1. Delete everything and start from scratch')
#             click.echo('2. Process the existing files')
#             click.echo('3. Exit')
#             choice = click.prompt('Enter your choice', type=int)
#             if choice == 1:
#                 click.echo('Deleting everything...')
#                 for file in folder_files:
#                     file.unlink()
#             elif choice == 2:
#                 click.echo('Processing existing files...')
#             elif choice == 3:
#                 click.echo('Exiting...')
#                 ctx.exit()
#             else:
#                 raise click.BadParameter(f'Invalid choice "{choice}".')
#     return abs_path


# DEMO CMD
# @click.command()
# @click.option('-it', '--iterations', type=int, default=100, show_default=True,
#     help='Number of iterations to run the algorithm.')
# @click.option('-o', '--output', 'output_folder',
#     # type=click.Path(exists=True),
#     type=click.Path(
#         # exists=True,
#         file_okay=False, dir_okay=True, resolve_path=True),
#     default='demo-output', show_default=True,
#     help='Location to save the generated images.',
#     callback=validate_and_normalize_path,
# )
def create_algo_runner(iterations=100, output_folder='gui-output-folder'):
    print("[DEBUG] output type: {}".format(type(output_folder)))

    current_directory = Path.cwd()

    termination_condition = 'max-iterations'

    content_img_file = current_directory / 'tests' / 'data' / 'canoe_water_w300-h225.jpg'
    style_img_file = current_directory / 'tests' / 'data' / 'blue-red_w300-h225.jpg'

    content_image, style_image = read_images(content_img_file, style_img_file)

    load_pretrained_model_functions()
    model_design = type('ModelDesign', (), {
        'pretrained_model': ModelHandlerFacility.create('vgg'),
        'network_design': NetworkDesign.from_default_vgg()
    })
    model_design.pretrained_model.load_model_layers()

    termination_condition = TerminationConditionAdapterFactory.create(
        termination_condition,
        iterations,  # maximun number of iterations to run the algorithm
    )

    print(f' -- Termination Condition: {termination_condition.termination_condition}')

    algorithm_parameters = AlogirthmParameters(
        content_image,
        style_image,
        termination_condition,
        output_folder,
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
        StylingObserver(Disk.save_image, convert_to_uint8, iterations)
    )
    return {
        'run': lambda: algorithm_runner.run(algorithm, model_design),
        'subscribe': lambda observer: algorithm_runner.progress_subject.add(observer),
    }
    # return algorithm_runner, lambda: algorithm_runner.run(algorithm, model_design)

    # algorithm_runner.run(algorithm, model_design)
