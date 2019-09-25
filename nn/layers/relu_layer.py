import numpy as np
from numba import njit, prange

from .layer import Layer


class ReLULayer(Layer):
    def __init__(self):
        super(ReLULayer, self).__init__()

    def forward(self, data):
        # TODO
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        return None


class ReLUNumbaLayer(Layer):
    def __init__(self):
        super(ReLUNumbaLayer, self).__init__()

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # TODO
        pass

    def forward(self, data):
        # Modify if you want
        output = self.forward_numba(data)
        return output

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO
        pass

    def backward(self, previous_partial_gradient):
        return None
