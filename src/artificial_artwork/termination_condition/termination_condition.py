from typing import Callable, Optional
import attr
from software_patterns import SubclassRegistry

from .termination_condition_interface import TerminationConditionInterface

# Future work: use an Abstract class while also inheriting from a generic
# interface
# class AbstractTerminationCondition(TerminationConditionInterface, Generic[T])


class TerminationFactoryMeta(SubclassRegistry[TerminationConditionInterface]):
    pass


class TerminationFactory(metaclass=TerminationFactoryMeta):
    pass

#####
# Implement Termination Conditions, by leveraging the Factory Method Pattern
# encapsulate any internal state, using one or more instance attribute(s) (ie see below 'max_iterations', 'min_improvement')
# Note: client code needs to know how many args each constructor supports

# Guide:
# 1. Use decorators as below
# 2. Inherit from TerminationConditionInterface, providing type for generic T (ie TerminationConditionInterface[int])
#    Type T corresponds to the 'progress' object in the `satisfied(self, progress: T)` method
# 4. Implement the `satisfied(self, iterations: T)` -> bool method, return True when the algo should stop

@attr.s
@TerminationFactory.register_as_subclass("max-iterations")
class MaxIterations(TerminationConditionInterface[int]):
    max_iterations: int = attr.ib()

    def satisfied(self, iterations: int) -> bool:
        return self.max_iterations <= iterations


@attr.s
@TerminationFactory.register_as_subclass("convergence")
class Convergence(TerminationConditionInterface[float]):
    min_improvement: float = attr.ib()

    def satisfied(self, last_loss_improvement: float) -> bool:
        return last_loss_improvement < self.min_improvement


@attr.s
@TerminationFactory.register_as_subclass("time-limit")
class TimeLimit(TerminationConditionInterface[float]):
    time_limit: float = attr.ib()

    def satisfied(self, duration: float) -> bool:
        return self.time_limit <= duration

StopSignal = Optional[Callable[[], bool]]
@attr.s
@TerminationFactory.register_as_subclass("stop-signal")
class RuntimeStopSignal(TerminationConditionInterface[StopSignal]):

    def satisfied(self, stop_signal: StopSignal) -> bool:
        if stop_signal is None:
            return False
        return stop_signal()


# Single factory/endpoint for creating TerminationCondition instances
class TerminationConditionFacility:
    class_registry: SubclassRegistry = TerminationFactory

    @classmethod
    def create(
        cls, termination_condition_type: str, *args, **kwargs
    ) -> TerminationConditionInterface:
        return cls.class_registry.create(termination_condition_type, *args, **kwargs)
