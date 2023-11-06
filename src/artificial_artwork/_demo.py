from pathlib import Path

from .algorithm import AlogirthmParameters, NSTAlgorithm
from .disk_operations import Disk
from .nst_image import convert_to_uint8, noisy
from .nst_tf_algorithm import NSTAlgorithmRunner
from .pretrained_model import ModelHandlerFacility
from .production_networks import NetworkDesign
from .styling_observer import StylingObserver
from .termination_condition_adapter_factory import TerminationConditionAdapterFactory
from .utils import load_pretrained_model_functions, read_images

this_file_directory = Path(__file__).parent
source_root_dir = this_file_directory.parent.parent


def create_algo_runner(
    iterations=100,
    output_folder="gui-output-folder",
    content_img_file=None,
    style_img_file=None,
):
    print("[DEBUG] output type: {}".format(type(output_folder)))

    termination_condition = "max-iterations"

    content_img_file = (
        content_img_file
        if content_img_file
        else source_root_dir / "tests" / "data" / "canoe_water_w300-h225.jpg"
    )
    style_img_file = (
        style_img_file
        if style_img_file
        else source_root_dir / "tests" / "data" / "blue-red_w300-h225.jpg"
    )

    content_image, style_image = read_images(content_img_file, style_img_file)

    load_pretrained_model_functions()  # ie import VGG ModelHandler implementation (to allow facility creating instances)
    model_design = type(
        "ModelDesign",
        (),
        {
            "pretrained_model": ModelHandlerFacility.create("vgg"),
            "network_design": NetworkDesign.from_default_vgg(),
        },
    )
    model_design.pretrained_model.load_model_layers()

    termination_condition = TerminationConditionAdapterFactory.create(
        termination_condition,
        iterations,  # maximun number of iterations to run the algorithm
    )

    print(f" -- Termination Condition: {termination_condition.termination_condition}")

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
        "run": lambda: algorithm_runner.run(algorithm, model_design),
        "subscribe": lambda observer: algorithm_runner.progress_subject.add(observer),
    }
    # return algorithm_runner, lambda: algorithm_runner.run(algorithm, model_design)

    # algorithm_runner.run(algorithm, model_design)
