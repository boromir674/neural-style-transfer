from numpy.typing import NDArray
import attr


@attr.s
class VggLayersGetter:
    vgg_layers: NDArray = attr.ib()  # iterable over layers data
    _vgg_layer_id_2_layer = attr.ib(init=False,
        default=attr.Factory(lambda self: {layer[0][0][0][0]: self.vgg_layers[index]
            for index, layer in enumerate(self.vgg_layers)}, takes_self=True))

    @property
    def id_2_layer(self):
        return self._vgg_layer_id_2_layer
