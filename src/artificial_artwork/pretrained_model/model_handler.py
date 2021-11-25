import os

from artificial_artwork.utils.subclass_registry import SubclassRegistry
from .model_handler_interface import ModelHandlerInterface
from .layers_getter import ModelReporter
from .model_routines import PretrainedModelRoutines


class Modelhandler(ModelHandlerInterface):
    def __init__(self):
        self._reporter = None
    
    @property
    def reporter(self):
        return self._reporter
    
    @reporter.setter
    def reporter(self, layers):
        self._reporter = ModelReporter(
            self.model_routines.get_layers_dict(layers),
            self.model_routines.get_weights
        )

    def load_model_layers(self):
        layers = self._load_model_layers()
        self.reporter = layers
        return layers

    def _load_model_layers(self):
        try:
            return self.model_routines.load_layers(os.environ[self.environment_variable])
        except KeyError as variable_not_found:
            raise NoImageModelSpesifiedError(self.model_load_exception_text) \
                from variable_not_found

class NoImageModelSpesifiedError(Exception): pass


class ModelHandlerFactory(metaclass=SubclassRegistry): pass
    

class ModelHandlerFacility:
    routines_interface = PretrainedModelRoutines
    handler_class = Modelhandler
    factory = ModelHandlerFactory

    @classmethod
    def create(cls, handler_type, *args, **kwargs) -> Modelhandler:
        return cls.factory.create(handler_type, *args, **kwargs)
