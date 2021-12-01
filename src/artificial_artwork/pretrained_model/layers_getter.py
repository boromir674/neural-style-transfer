from typing import Callable, Tuple, Dict
from numpy.typing import NDArray
from attr import define


@define
class ModelReporter:
    _layer_id_2_layer: Dict[str, NDArray]
    _weights_extractor: Callable[[NDArray], Tuple[NDArray, NDArray]]

    def layer(self, layer_id: str) -> NDArray:
        return self._layer_id_2_layer[layer_id]

    def weights(self, layer: NDArray) -> Tuple[NDArray, NDArray]:
        return self._weights_extractor(layer)

    def get_weights(self, layer_id: str) -> Tuple[NDArray, NDArray]:
        return self.weights(self.layer(layer_id))
