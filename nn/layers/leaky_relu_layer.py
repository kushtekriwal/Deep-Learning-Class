from numba import njit, prange

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope=0.1):
        super(LeakyReLULayer, self).__init__()
        self.slope = slope

    def forward(self, data):
        # TODO
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        return None
