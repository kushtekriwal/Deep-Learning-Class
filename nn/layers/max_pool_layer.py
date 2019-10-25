import numbers

import numpy as np
from numba import njit, prange

from .layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, k, stride, padding):
        # data is N x C x H x W
        # TODO
        (batchsize, inchannels, height, width) = list(data.shape)
        outheight = int(((height - k) / stride) + 1)
        outwidth = int((width - k) / stride + 1)
        output = np.zeros((batchsize, inchannels, outheight, outwidth), dtype=np.float32)
        for image in prange(batchsize):
            for channel in prange(inchannels):
                for h in prange(outheight):
                    for w in prange(outwidth):
                        vert_start = h * stride
                        vert_end = vert_start + k
                        horiz_start = w * stride
                        horiz_end = horiz_start + k
        
                        subset = data[image][channel,vert_start:vert_end,horiz_start:horiz_end]
                        output[image, channel, h, w] = np.max(subset)
        return output

    def forward(self, data):
        # TODO
        data_pad = np.pad(data, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0,0))
        output = MaxPoolLayer.forward_numba(data_pad, self.kernel_size, self.stride, self.padding)
        self.data = data
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, dA_prev, dA_prev_pad, A_prev_pad, padding, stride, data, k):
        # data is N x C x H x W
        # TODO
        (batchsize, channels, height, width) = list(data.shape)
        outheight = int(((height - k + 2*padding) / stride) + 1)
        outwidth = int(((width - k + 2*padding) / stride) + 1)

        for i in range(batchsize):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for c in range(channels):
                for h in range(outheight):
                    for w in range(outwidth):
                        vert_start = h * stride
                        vert_end = vert_start + k
                        horiz_start = w * stride
                        horiz_end = horiz_start + k

                        a_slice = a_prev_pad[c, vert_start:vert_end, horiz_start:horiz_end]
                        mask = (a_slice == np.max(a_slice))
                        da_prev_pad[c, vert_start:vert_end, horiz_start:horiz_end] += mask * previous_grad[i, c, h, w]
            if padding == 0:
                dA_prev[i,:,:,:] = da_prev_pad[:, :, :]
            else:
                dA_prev[i,:,:,:] = da_prev_pad[:, padding:-padding, padding:-padding]                    
        return dA_prev

    def backward(self, previous_partial_gradient):
        # TODO
        (batchsize, inchannels, height, width) = list(self.data.shape)

        dA_prev = np.zeros((batchsize, inchannels, height, width), dtype=np.float32) 
        dA_prev_pad = np.pad(dA_prev, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0,0))
        A_prev_pad = np.pad(self.data, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0,0))
        
        dA_prev = MaxPoolLayer.backward_numba(previous_partial_gradient, dA_prev, dA_prev_pad, A_prev_pad, self.padding, self.stride, self.data, self.kernel_size)
        return dA_prev

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))
