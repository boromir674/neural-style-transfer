from .termination_condition_adapter import TerminationConditionAdapterClassFactory
from .termination_condition import TerminationConditionFacility


class TerminationConditionAdapterFactory:

    @classmethod
    def create(cls, adapter_type: str, *args):
        dynamic_class = TerminationConditionAdapterClassFactory.create(adapter_type)
        termination_condition = TerminationConditionFacility.create(adapter_type, *args)
        return dynamic_class(termination_condition)
