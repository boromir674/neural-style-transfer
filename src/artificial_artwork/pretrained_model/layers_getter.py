from typing import Callable, Tuple
from numpy.typing import NDArray
import attr


# @attr.s
# class VggLayersGetter:
#     vgg_layers: NDArray = attr.ib()  # iterable over layers data
#     _vgg_layer_id_2_layer = attr.ib(init=False,
#         default=attr.Factory(lambda self: {layer[0][0][0][0]: self.vgg_layers[index]
#             for index, layer in enumerate(self.vgg_layers)}, takes_self=True))

#     @property
#     def id_2_layer(self):
#         return self._vgg_layer_id_2_layer


@attr.s
class ModelReporter:
    _layer_id_2_layer = attr.ib(
        converter=lambda layers: {layer[0][0][0][0]: layers[index]
            for index, layer in enumerate(layers)})
    _weights_extractor: Callable[[NDArray], Tuple[NDArray, NDArray]] = attr.ib()
    
    def layer(self, layer_id: str) -> NDArray:
        return self._layer_id_2_layer[layer_id]

    def weights(self, layer: NDArray) -> Tuple[NDArray, NDArray]:
        return self._weights_extractor(layer)

    def get_weights(self, layer_id: str) -> Tuple[NDArray, NDArray]:
        return self.weights(self.layer(layer_id))
