from abc import ABC
import attr


class AlgorithmInterface(ABC):
    """An algorithm is a series of execution steps, aiming to solve a problem.

    This interface provides a 'run' method that sequentially runs the
    algorithm's steps.
    """
    def run(self, *args, **kwargs) -> None:
        """Run the algorithm."""
        raise NotImplementedError


class IterativeAlgorithm(AlgorithmInterface):

    def run(self, *args, **kwargs) -> None:
        pass


class LearningAlgorithm(IterativeAlgorithm):

    def run(self, *args, **kwargs) -> None:
        pass

    def compute_cost(self, *args, **kwargs) -> float:
        raise NotImplementedError


@attr.s
class NSTAlgorithm(IterativeAlgorithm):
    parameters = attr.ib()
    image_config = attr.ib()

    def run(self, *args, **kwargs) -> None:
        return super().run(*args, **kwargs)


from .style_layer_selector import NSTLayersSelection


@attr.s
class AlogirthmParameters:
    # TODO remove content and style images and output_path
    # retain only algorithm parameters (variables governing how the algo will behave)
    # from the algo input (runtime objects that are the INPUT to the algo)
    content_image = attr.ib()
    style_image = attr.ib()
    cv_model = attr.ib()
    style_layers = attr.ib(converter=NSTLayersSelection.from_tuples)
    termination_condition = attr.ib()
    output_path = attr.ib()
