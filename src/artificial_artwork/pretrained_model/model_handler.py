import os
from typing import Protocol, Tuple

from numpy.typing import NDArray
from software_patterns import SubclassRegistry

from .layers_getter import ModelReporter
from .model_handler_interface import ModelHandlerInterface


class ReporterProtocol(Protocol):
    def get_weights(self, layer_id: str) -> Tuple[NDArray, NDArray]:
        ...


class Modelhandler(ModelHandlerInterface):
    """Handle

    Args:
        ModelHandlerInterface ([type]): [description]

    Raises:
        NoImageModelSpesifiedError: [description]

    Returns:
        [type]: [description]
    """

    _reporter: ReporterProtocol

    def __init__(self):
        self._reporter = None

    @property
    def reporter(self) -> ReporterProtocol:
        """Reporter getter, which can return Layer Weights given a layer (str) ID

        Args:
            layers ([type]): [description]
        """
        return self._reporter

    @reporter.setter
    def reporter(self, layers) -> None:
        self._reporter = self._create_reporter(layers)

    def _create_reporter(self, layers: NDArray) -> ReporterProtocol:
        return ModelReporter(
            self.model_routines.get_layers_dict(layers), self.model_routines.get_weights
        )

    def load_model_layers(self) -> NDArray:
        layers = self._load_model_layers()
        self._reporter = self._create_reporter(layers)
        return layers

    def _load_model_layers(self) -> NDArray:
        try:
            return self.model_routines.load_layers(os.environ[self.environment_variable])
        except KeyError as variable_not_found:
            raise NoImageModelSpesifiedError(
                self.model_load_exception_text
            ) from variable_not_found


class NoImageModelSpesifiedError(Exception):
    pass


class ModelHandlerFactoryMeta(SubclassRegistry[Modelhandler]):
    pass


class ModelHandlerFactory(metaclass=ModelHandlerFactoryMeta):
    pass


class ModelHandlerFacility:
    # routines_interface: type = PretrainedModelRoutines
    handler_class: type = Modelhandler
    factory = ModelHandlerFactory

    @classmethod
    def create(cls, handler_type, *args, **kwargs) -> Modelhandler:
        return cls.factory.create(handler_type, *args, **kwargs)
