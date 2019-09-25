import numpy as np

from nn import Parameter
from .layer import Layer


class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.weight = Parameter(np.zeros((input_size, output_size), dtype=np.float32))
        self.bias = Parameter(np.zeros((1, output_size), dtype=np.float32))

    def forward(self, data):
        # TODO do the linear layer
        return None

    def backward(self, previous_partial_gradient):
        # TODO do the backward step
        return None

    def selfstr(self):
        return str(self.weight.data.shape)
