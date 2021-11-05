"""Implementation of the object pool"""
from typing import Dict, Generic, TypeVar, Any
import types
import attr


__all__ = ['ObjectsPool']


# Define objects to use for type annotation and (static) type checking with mypy

T = TypeVar('T')


def _validate_build_hash_callback(self, attribute, build_hash_callback):
    def method(_self, *args, **kwargs):
        return build_hash_callback(*args, **kwargs)
    setattr(self, attribute.name, types.MethodType(method, self))


@attr.s
class ObjectsPool(Generic[T]):
    """Class of objects that are able to return a reference to an object upon request.

    Whenever an object is requested, it is checked whether it exists in the pool.
    Then if it exists, a reference is returned, otherwise a new object is
    constructed (given the provided callable) and its reference is returned.

    Arguments:
        constructor (callable): able to construct the object given arguments
        objects (dict): the data structure representing the object pool
    """
    @staticmethod
    def __build_hash(*args: Any, **kwargs: Any) -> int:
        r"""Construct a hash out of the input \*args and \*\*kwargs."""
        return hash('-'.join([str(_) for _ in args] + [f'{key}={str(value)}' for key, value in kwargs.items()]))

    constructor = attr.ib()
    _build_hash = attr.ib(validator=_validate_build_hash_callback)
    _objects: Dict[int, T] = attr.ib(default={})

    def get_object(self, *args: Any, **kwargs: Any) -> T:
        r"""Request an object from the pool.

        Get or create an object given the input parameters. Existence in the pool is done using the
        python-build-in hash function. The input \*args and \*\*kwargs serve as
        input in the hash function to create unique keys with which to "query" the object pool.

        Returns:
            object: the reference to the object that corresponds to the input
            arguments, regardless of whether it was found in the pool or not
        """
        key = self._build_hash(*args, **kwargs)
        if key not in self._objects:
            self._objects[key] = self.constructor(*args, **kwargs)
        return self._objects[key]

    @classmethod
    def new_empty(cls, constructor, build_hash=None):
        if build_hash is None:
            build_hash = ObjectsPool.__build_hash
        return ObjectsPool(constructor, build_hash)
