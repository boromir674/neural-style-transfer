from typing import Union, TypeVar, List


Tensor = TypeVar('Tensor')
ksize_type = Union[int, List[int]]


def avg_pool(layer: Tensor, ksize: ksize_type, strides: ksize_type, padding: str) -> Tensor: ...
