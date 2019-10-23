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
        relu_b = previous_partial_gradient.copy()
        relu_b[self.data <= 0] = 0
        relu_b[self.data > 0] = 1
        return relu_b  


class ReLUNumbaLayer(Layer):
    def __init__(self, parent=None):
        super(ReLUNumbaLayer, self).__init__(parent)
        self.data = None

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data):
        # TODO Helper function for computing ReLU
        shape = data.shape
        relu_f = data.flatten()
        for idx, i in enumerate(relu_f):
            relu_f[idx] = np.maximum(i, 0)
        relu_f.reshape(shape)
        return relu_f

    def forward(self, data):
        # TODO
        self.data = data
        output = self.forward_numba(self.data)
        return output        

    @staticmethod
    @njit(parallel=True, cache=True)
    def backward_numba(data, grad):
        # TODO Helper function for computing ReLU gradients
        shape = grad.shape
        out_grad = grad.flatten()
        for idx, i in enumerate(out_grad):
            if data[idx] <= 0:
                out_grad[idx] = 0
            else:
                out_grad[idx] = 1
        return out_grad.reshape(shape)

    def backward(self, previous_partial_gradient):
        # TODO
        output = self.backward_numba(self.data, previous_partial_gradient)
        return output
