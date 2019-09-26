import numpy as np
from numba import njit, prange

from nn import Parameter
from .layer import Layer


class PReLULayer(Layer):
    def __init__(self, size: int, initial_slope: float = 0.1):
        super(PReLULayer, self).__init__()
        self.slope = Parameter(np.full(size, initial_slope))

    def initialize(self, *args, **kwargs):
        # No need to modify
        pass

    def forward(self, data):
        # TODO
        return None

    def backward(self, previous_partial_gradient):
        # TODO
        return None
