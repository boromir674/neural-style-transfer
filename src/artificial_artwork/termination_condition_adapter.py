from abc import ABC
from typing import Callable, Protocol, Dict, Any
from types import MethodType
from .termination_condition.termination_condition_interface import TerminationConditionInterface

from .utils.memoize import ObjectsPool


class MetricsCapable(Protocol):
    metrics: Dict[str, Any]

__all__ = ['TerminationConditionAdapterFactory']


class AbstractTerminationConditionAdapter(ABC):
    termination_condition: TerminationConditionInterface
    update: Callable[[MetricsCapable], None]

    def __new__(cls, termination_condition, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance.termination_condition = termination_condition
        instance.update = 1
        # instance.update = MethodType(cls._update_callback(cls.adapter_type), instance.update)
        instance.runtime_state = cls._initial_state_callback(cls.adapter_type)()
        return instance

    def __init__(self, *args, **kwargs):
        # TODO move all code in __new__
        self.update = MethodType(type(self)._update_callback(type(self).adapter_type), self)
        # setattr(self, attribute.name, types.MethodType(method, self))

    @classmethod
    def _update_callback(cls, type: str):
        def update(self, *args, **kwargs) -> None:
            self.runtime_state = args[0].state.metrics[cls.mapping[type]['key_name']]
        return update

    @classmethod
    def _initial_state_callback(cls, type: str):
        def get_initial_state():
            return cls.mapping[type]['state']
        return get_initial_state

    @property
    def satisfied(self):
        return self.termination_condition.satisfied(self.runtime_state)

 
# Define Metaclass
class TerminationConditionAdapterType(type):

    def __new__(mcs, *args, **kwargs):
        # termination_condition_adapter_class = super().__new__(mcs, 'TerminationConditionAdapter', (AbstractTerminationConditionAdapter,), {})
        termination_condition_adapter_class = type('TerminationConditionAdapter', (AbstractTerminationConditionAdapter,), {})
        termination_condition_adapter_class.adapter_type = args[0]
        
        # The (outer) keys are usable by client code to select termination condition
        # Each (inner) 'key_name' points to the name to use to query the subject dict
        # Each 'state' key points to a value that should be used to initialize
        # the 'runtime_state' attribute [per termination condition (adapter)]
        termination_condition_adapter_class.mapping = {
            'max-iterations': {'key_name': 'iterations', 'state': 0},
            'convergence': {'key_name': 'cost', 'state': float('inf')},
            'time-limit': {'key_name': 'duration', 'state': 0},
        }
        return termination_condition_adapter_class

    # Investigate usage of __init__ to verify the above behaviour can be replicated with __init__

class TerminationConditionAdapterClassFactory:
    """Acts as a proxy to the the 'class maker' function by returning a memoized class."""
    classes_pool = ObjectsPool.new_empty(TerminationConditionAdapterType)

    @classmethod
    def create(cls, adapter_type: str):
        return cls.classes_pool.get_object(adapter_type)


class TerminationConditionAdapterFactory:
    
    @classmethod
    def create(cls, adapter_type: str, *args, **kwargs):
        dynamic_class = TerminationConditionAdapterClassFactory.create(adapter_type)
        return dynamic_class(*args, **kwargs)
