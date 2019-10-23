from numba import njit, prange
import numpy as np

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = 0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope
        self.data = None

    def forward(self, data):
        # TODO
        shape = data.shape
        leaky = data.flatten()
        leaky[leaky <= 0] = self.slope * leaky[leaky <= 0]
        self.data = data
        return leaky.reshape(shape)

    def backward(self, previous_partial_gradient):
        # TODO
        leaky_b = previous_partial_gradient.copy()
        leaky_b[self.data <= 0] = self.slope
        leaky_b[self.data > 0] = 1
        return leaky_b  
