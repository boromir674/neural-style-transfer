from abc import ABC
from types import MethodType
from typing import Any, Callable, Dict, Protocol

from software_patterns import ObjectsPool


class MetricsCapable(Protocol):
    metrics: Dict[str, Any]


class TerminationConditionProtocol(Protocol):
    def satisfied(self, progress: Any) -> bool:
        ...


class AbstractTerminationConditionAdapter(ABC):
    termination_condition: TerminationConditionProtocol
    update: Callable[[MetricsCapable], None]
    runtime_state: Any
    mapping: Dict[str, Dict]

    def __new__(cls, termination_condition, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        instance.termination_condition = termination_condition
        instance.runtime_state = cls._initial_state_callback(cls.adapter_type)()
        instance.update = MethodType(
            type(instance)._update_callback(type(instance).adapter_type), instance
        )
        return instance

    @classmethod
    def _update_callback(cls, termination_type: str):
        def update(self, *args, **kwargs) -> None:
            self.runtime_state = args[0].state.metrics[
                cls.mapping[termination_type]["key_name"]
            ]

        return update

    @classmethod
    def _initial_state_callback(cls, termination_type: str):
        def get_initial_state():
            return cls.mapping[termination_type]["state"]

        return get_initial_state

    @property
    def satisfied(self):
        return self.termination_condition.satisfied(self.runtime_state)


class BaseTerminationConditionAdapter:
    pass


def new_class(adapter_type: str):
    return type(
        "TerminationConditionAdapterC",
        (AbstractTerminationConditionAdapter,),
        {
            "adapter_type": adapter_type,
            "mapping": {
                "max-iterations": {"key_name": "iterations", "state": 0},
                "convergence": {"key_name": "cost", "state": float("inf")},
                "time-limit": {"key_name": "duration", "state": 0},
            },
        },
    )


class TerminationConditionAdapterClassFactory:
    """Acts as a proxy to the the 'class maker' function by returning a memoized class."""

    classes_pool = ObjectsPool(new_class)

    @classmethod
    def create(cls, adapter_type: str):
        return cls.classes_pool.get_object(adapter_type)
