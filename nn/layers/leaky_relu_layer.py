from numba import njit, prange
import numpy as np

from .layer import Layer


class LeakyReLULayer(Layer):
    def __init__(self, slope: float = 0.1, parent=None):
        super(LeakyReLULayer, self).__init__(parent)
        self.slope = slope
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, slope):
        data_length = list(data.shape)
        data_length = data_length[0]
        for i in range(data_length):
            if (data[i] <= 0):
                data[i] = slope * data[i]
        return data

    def forward(self, data):
        self.data = data
        input_shape = data.shape
        flattened_inp = data.flatten()
        flattened_oup = self.forward_numba(flattened_inp, self.slope)
        output = flattened_oup.reshape(input_shape)
        return output  

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad, slope):
        grad_length = list(grad.shape)
        grad_length = grad_length[0]
        for i in range(grad_length):
            if (data[i] <= 0):
                grad[i] = slope
        return grad

    def backward(self, previous_partial_gradient):
        # TODO
        prev_shape = previous_partial_gradient.shape
        flat_data = self.data.flatten()
        flat_prev = previous_partial_gradient.flatten()
        flat_gradient = self.backward_numba(flat_data, flat_prev, self.slope)
        gradient = flat_gradient.reshape(prev_shape)
        return gradient
