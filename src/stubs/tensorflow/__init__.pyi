from typing import List, Protocol, TypeVar, Union

T = TypeVar("T")

# future work: change T to indicate either numpy arrays or tensorflow tensors
def matmul(volume_1: T, volume_2: T) -> T: ...

Tensor = TypeVar("Tensor")

def transpose(volume_1: Tensor) -> Tensor: ...
def constant(tensor: Tensor) -> Tensor: ...

ksize_type = Union[int, List[int]]

class nn(Protocol):
    @staticmethod
    def avg_pool(
        layer: Tensor, ksize: ksize_type, strides: ksize_type, padding: str
    ) -> Tensor: ...
    @staticmethod
    def relu(layer: Tensor) -> Tensor: ...
    @staticmethod
    def conv2d(layer: Tensor, filter: Tensor, strides: ksize_type, padding: str) -> Tensor: ...

class TrainAPI(Protocol):
    @staticmethod
    def AdamOptimizer(learning_rate: float):
        pass

class TensorflowV1Api(Protocol):
    nn: nn
    train: TrainAPI

class compat(Protocol):
    v1: TensorflowV1Api
