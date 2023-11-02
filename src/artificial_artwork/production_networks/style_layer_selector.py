from typing import Iterable, List, Protocol, Tuple

import attr


@attr.s
class NSTStyleLayer:
    id: str = attr.ib()
    coefficient: float = attr.ib()

    @coefficient.validator
    def validate_weight(self, attribute, value):
        if value <= 0 or 1 < value:
            raise ValueError(f"Coefficient must be a number between 0 and 1. Given {value}")

    neurons = attr.ib(default=None)


def validate_layers(layers):
    if abs(weights_sum := sum(coefs := [layer.coefficient for layer in layers]) - 1) > 1e-6:
        raise ValueError(
            f'Coefficients of selected Style Layers do not sum to 1: sum({", ".join((str(_) for _ in coefs))}) = {weights_sum}'
        )
    if len(set(layer_ids := [layer.id for layer in layers])) != len(layers):
        raise ValueError(
            f'Duplicate Style Layers found in the selection: [{", ".join(layer_ids)}]'
        )


class StyleLayerSelectionProtocol(Protocol):
    layers: List[NSTStyleLayer]


@attr.s
class NSTLayersSelection:
    _layers: List[NSTStyleLayer] = attr.ib(
        validator=lambda self, attribute, layers: validate_layers(layers)
    )

    @classmethod
    def from_tuples(cls, layers: Iterable[Tuple[str, float]]) -> StyleLayerSelectionProtocol:
        return NSTLayersSelection([NSTStyleLayer(*layer) for layer in layers])
        # return NSTLayersSelection([NSTStyleLayer(layer[0], layer[1]) for layer in layers])

    @property
    def layers(self) -> List[NSTStyleLayer]:
        return self._layers

    @layers.setter
    def layers(self, layers) -> None:
        """Set the Style Layers selection.

        Args:
            layers ([type]): [description]

        Raises:
            ValueError: if layers' coefficients don't sum to 1 or with
        duplicate layers ids
        """
        validate_layers(layers)
        self._layers = layers

    def __getitem__(self, index) -> NSTStyleLayer:
        return self._layers[index]

    def __iter__(self) -> Iterable[Tuple[str, NSTStyleLayer]]:
        return iter(((layer.id, layer) for layer in self._layers))
