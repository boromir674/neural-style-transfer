
from abc import ABCMeta, abstractmethod

class NeuralTransfer(metaclass=ABCMeta):

    @abstractmethod
    def content_cost(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def style_cost(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def total_cost(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def fine_tune(self, *args, **kwargs):
        raise NotImplementedError
