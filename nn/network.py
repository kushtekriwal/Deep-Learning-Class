from abc import ABC

from nn.layers import Layer
from nn.layers.losses import LossLayer


class Network(Layer, ABC):
    def __init__(self, network: Layer, loss_layer: LossLayer):
        super(Network, self).__init__()
        self.network: Layer = network
        self.loss_layer: LossLayer = loss_layer

    def loss(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def backward(self) -> None:
        gradient = self.loss_layer.backward()
        self.network.backward(gradient)
