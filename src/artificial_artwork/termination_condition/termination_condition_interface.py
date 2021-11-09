from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')


class TerminationConditionInterface(ABC, Generic[T]):
    """A condition that evaluates to True or False.

    If True it should indicate that something should now terminate.
    """
    @abstractmethod
    def satisfied(self, progress: T) -> bool:
        """Check if the termination condition is True.

        Args:
            progress ([type]): [description]

        Returns:
            bool: True if the termination condition is satisfied, else False
        """
        raise NotImplementedError
