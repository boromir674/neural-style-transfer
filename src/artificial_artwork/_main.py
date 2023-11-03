"""Bridges the Backend code with the CLI's main (aka run) cmd
"""
import os

from .algorithm import AlogirthmParameters, NSTAlgorithm
from .disk_operations import Disk
from .nst_image import convert_to_uint8
from .nst_tf_algorithm import NSTAlgorithmRunner
from .pretrained_model import ModelHandlerFacility
from .production_networks import NetworkDesign
from .styling_observer import StylingObserver
from .termination_condition_adapter_factory import TerminationConditionAdapterFactory
from .utils import load_pretrained_model_functions, read_images

this_file_location = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))


__all__ = ["create_algo_runner"]


def create_algo_runner(
    iterations=100, output_folder="gui-output-folder", noisy_ratio=0.6  # ratio
):
    termination_condition = _create_termination_condition(iterations)

    algorithm_runner = _create_algo_runner(termination_condition, noisy_ratio=noisy_ratio)

    def run(content_image, style_image):
        algorithm = _read_algorithm_input(
            content_image, style_image, termination_condition, output_folder
        )
        model_design = _load_algorithm_architecture()

        algorithm_runner.run(algorithm, model_design)

    return {
        "run": run,
        "subscribe": lambda observer: algorithm_runner.progress_subject.add(observer),
    }


def _create_algo_runner(termination_condition, noisy_ratio=0.6):
    import tensorflow as tf

    from artificial_artwork.image import noisy
    from artificial_artwork.tf_session_runner import (
        TensorflowSessionRunner,
        TensorflowSessionRunnerSubject,
    )

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    tf_session_wrapper = TensorflowSessionRunner(
        TensorflowSessionRunnerSubject(tf.compat.v1.InteractiveSession())
    )
    # session_runner = TensorflowSessionRunner.with_default_graph_reset()
    algorithm_runner = NSTAlgorithmRunner(
        tf_session_wrapper,
        lambda matrix: noisy(matrix, noisy_ratio),
    )
    # algorithm_runner = NSTAlgorithmRunner.default(
    #     lambda matrix: noisy(matrix, noisy_ratio),
    # )
    # Subscribe the termination_condition object so that ir receives updates
    # whenever the runner broadcasts updates.
    # The NST Algorithm Runner broadcasts updates on a steady frequency during
    # the run. It always broadcats on First and Last Iteration. For example,
    # if the run is 100 iterations, it will broadcast on iterations
    # 0, 20, 40, 60, 80, 100
    # Each broadcast is an event 'carrying' a progress object, which is a python
    # Dict
    # For more on the expected keys and values of the progress Dict see the
    # '_progress' instance method defined in the
    # artificial_artwork.nst_tf_algorithm.py > NSTAlgorithmRunner class

    algorithm_runner.progress_subject.add(
        termination_condition,
    )
    # Subscribe Persistance so that we keep snaphosts of the generated images in the disk
    algorithm_runner.persistance_subject.add(
        StylingObserver(
            Disk.save_image,
            convert_to_uint8,
            termination_condition.termination_condition.max_iterations,
        )
    )
    return algorithm_runner


DEFAULT_TERMINATION_CONDITION = "max-iterations"


def _create_termination_condition(nb_iterations_to_perform):
    _ = TerminationConditionAdapterFactory.create(
        DEFAULT_TERMINATION_CONDITION,
        nb_iterations_to_perform,
    )
    print(f" -- Termination Condition: {_.termination_condition}")
    return _


def _load_algorithm_architecture():
    load_pretrained_model_functions()
    model_design = type(
        "ModelDesign",
        (),
        {
            "pretrained_model": ModelHandlerFacility.create("vgg"),
            "network_design": NetworkDesign.from_default_vgg(),
        },
    )
    model_design.pretrained_model.load_model_layers()
    return model_design


def _read_algorithm_input(content_image, style_image, termination_condition, location):
    # Read Images given their file paths in the disk (filesystem)
    content_image, style_image = read_images(content_image, style_image)

    # Compute Termination Condition, given input number of iterations to perform
    # The number of iterations is the number the image will pass through the
    # network. The more iterations the more the Style is applied.
    #
    # The number of iterations is not the number of times the network
    # will be trained. The network is trained only once, and the image is
    # passed through it multiple times.

    return NSTAlgorithm(
        AlogirthmParameters(
            content_image,
            style_image,
            termination_condition,
            location,
        )
    )
