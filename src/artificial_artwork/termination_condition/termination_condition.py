import attr

from artificial_artwork.utils.subclass_registry import SubclassRegistry
from .termination_condition_interface import TerminationConditionInterface

# TODO: learn how to use the Abstract class implementing a generic interface
# and then inherit from the Abstract class
# T = TypeVar('T')

# class AbstractTerminationCondition(TerminationConditionInterface, Generic[T]): pass


class TerminationFactory(metaclass=SubclassRegistry):
    pass


@attr.s
@TerminationFactory.register_as_subclass('max-iterations')
class MaxIterations(TerminationConditionInterface[int]):
    max_iterations: int = attr.ib()

    def satisfied(self, iterations: int) -> bool:
        return self.max_iterations <= iterations

@attr.s
@TerminationFactory.register_as_subclass('convergence')
class Convergence(TerminationConditionInterface[float]):
    min_improvement: float = attr.ib()

    def satisfied(self, last_loss_improvement: float) -> bool:
        return last_loss_improvement < self.min_improvement

@attr.s
@TerminationFactory.register_as_subclass('time-limit')
class TimeLimit(TerminationConditionInterface[float]):
    time_limit: float = attr.ib()

    def satisfied(self, duration: float) -> bool:
        return self.time_limit <= duration


class TerminationConditionFacility:
    class_registry: SubclassRegistry = TerminationFactory

    @classmethod
    def create(cls, termination_condition_type: str, *args, **kwargs) -> TerminationConditionInterface:
        return cls.class_registry.create(termination_condition_type, *args, **kwargs)
