"""Bridges the Backend code with the CLI's main (aka run) cmd
"""

import logging
import os
import typing as t

from .algorithm import AlogirthmParameters, NSTAlgorithm
from .disk_operations import Disk
from .nst_image import convert_to_uint8
from .nst_tf_algorithm import NSTAlgorithmRunner
from .pretrained_model import ModelHandlerFacility
from .production_networks import NetworkDesign
from .styling_observer import StylingObserver
from .termination_condition_adapter_factory import TerminationConditionAdapterFactory
from .utils import load_pretrained_model_functions, read_images

logger = logging.getLogger(__name__)
this_file_location = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))


__all__ = ["create_algo_runner"]

# CONSTANTS
DEFAULT_TERMINATION_CONDITION = "max-iterations"


def create_algo_runner(
    iterations=100, output_folder="gui-output-folder",
    noisy_ratio=0.6,  # ratio
    auto_resize_style=False,
):
    # Max Epoch Termination Condition: stop when Iterative Algorith reaches a certain number of iterations
    max_epochs_termination_condition = TerminationConditionAdapterFactory.create(
        DEFAULT_TERMINATION_CONDITION,
        iterations,
    )
    logger.info("Termination Condition: %s %s", DEFAULT_TERMINATION_CONDITION, iterations)

    # Stop Signal Termination Condition: stop when a (user) signal is received
    stop_signal_termination_condition = TerminationConditionAdapterFactory.create(
        "stop-signal",
    )
    logger.info("Termination Condition: %s", "stop-signal")

    termination_conditions=[max_epochs_termination_condition, stop_signal_termination_condition]

    # Subscribe Termination Conditions to Progress Event Updates to allow them to evaluate if satisfied
    algorithm_runner = _create_algo_runner(termination_conditions, noisy_ratio=noisy_ratio)

    def run(content_image, style_image, stop_signal: t.Optional[t.Callable[[], bool]] = None):
        algorithm = _read_algorithm_input(
            # Images CONTENT and STYLE
            content_image, style_image,
            # pass Termination Conditions to allow algo to query for finished check
            termination_conditions,  # ie Max Epochs, Convergence Reached, User Stop Signal
            output_folder,
            auto_resize_style=auto_resize_style,
        )
        # Pretrained Model Weights (NN Layers) and get_weights method
        model_design = _load_algorithm_architecture()

        algorithm_runner.run(algorithm, model_design, stop_signal=stop_signal)

    return {
        "run": run,
        "subscribe_to_progress": lambda observer: algorithm_runner.progress_subject.add(observer),
        "subscribe_to_running_flag": lambda observer: algorithm_runner.running_flag_subject.add(observer),
    }


def _create_algo_runner(termination_conditions, noisy_ratio=0.6):
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
        lambda matrix: noisy(matrix, noisy_ratio),  # TODO: retire, since it is only used when demo CLI command is called
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

    # Subscribe the TerminationCondition instances to progress updates from the NST Algo Runner
    algorithm_runner.progress_subject.add(
        *termination_conditions
    )

    # Subscribe Object that persists snapshots of the Generated Image, saving on disk
    algorithm_runner.persistance_subject.add(
        StylingObserver(
            Disk.save_image,
            convert_to_uint8,
            termination_conditions[0].termination_condition.max_iterations,
        )
    )
    return algorithm_runner


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
    # Load pretrained model weights (NN layers learned parameters) from disk, into memory
    # expose pretrained_model.reporter.get_weights(layer_id: str) -> Tuple[NDArray, NDArray]
    model_design.pretrained_model.load_model_layers()
    return model_design


def _read_algorithm_input(content_image, style_image, termination_conditions, location, auto_resize_style=False):

    # Read Images given their file paths in the disk (filesystem)
    content_image, style_image = read_images(content_image, style_image, auto_resize_style=auto_resize_style)

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
            termination_conditions,
            location,
        )
    )
