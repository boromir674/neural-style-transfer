from abc import ABC, abstractmethod

from .model_routines import PretrainedModelRoutines


class ModelHandlerInterface(ABC):

    @property
    @abstractmethod
    def environment_variable(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_routines(self) -> PretrainedModelRoutines:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_load_exception_text(self) -> str:
        raise NotImplementedError
