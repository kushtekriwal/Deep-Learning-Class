import numpy as np
from numba import njit, prange

from .layer import Layer


class ReLULayer(Layer):
    def __init__(self, parent=None):
        super(ReLULayer, self).__init__(parent)
        self.data = None

    def forward(self, data):
        # TODO
        relu_f = np.maximum(data, 0)
        self.data = data
        return relu_f

    def backward(self, previous_partial_gradient):
        # TODO
        #data_shape = list(self.data.shape)
        #data = self.data.reshape(data_shape[0], -1)
        relu_b = np.multiply(previous_partial_gradient, (self.data>0))
        return relu_b  


class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # TODO Helper function for computing ReLU
        data_length = list(data.shape)
        data_length = data_length[0]
        for i in range(data_length):
            if (data[i] <= 0):
                data[i] = 0
        return data

    def forward(self, data):
        # TODO 
        self.data = data
        input_shape = data.shape
        flattened_inp = data.flatten()
        flattened_oup = self.forward_numba(flattened_inp)
        output = flattened_oup.reshape(input_shape)
        return output        

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO Helper function for computing ReLU gradients
        grad_length = list(grad.shape)
        grad_length = grad_length[0]
        for i in range(grad_length):
            if (data[i] <= 0):
                grad[i] = 0
        return grad

    def backward(self, previous_partial_gradient):
        # TODO
        prev_shape = previous_partial_gradient.shape
        flat_data = self.data.flatten()
        flat_prev = previous_partial_gradient.flatten()
        flat_gradient = self.backward_numba(flat_data, flat_prev)
        gradient = flat_gradient.reshape(prev_shape)
        return gradient
