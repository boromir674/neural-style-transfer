from numpy.typing import NDArray
import attr

from .vgg_architecture import LAYERS


@attr.s
class VggLayersGetter:
    vgg_layers: NDArray = attr.ib()
    _vgg_layer_id_2_layer = attr.ib(init=False,
        default=attr.Factory(lambda self: {layer_id: self.vgg_layers[index] for index, layer_id in enumerate(LAYERS)}, takes_self=True))
    
    @property
    def id_2_layer(self):
        return self._vgg_layer_id_2_layer
