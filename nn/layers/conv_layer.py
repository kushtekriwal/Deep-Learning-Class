from typing import Optional, Callable
import numpy as np

from numba import njit, prange

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.data = None
        #self.fprop = None
        self.initialize()

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data, weights, bias, stride, padding):
        # TODO
        (batchsize, _, height, width) = list(data.shape)
        (_, outchannels, k, k) = list(weights.shape)
        outheight = int(((height - k) / stride) + 1)
        outwidth = int((width - k) / stride + 1)
        output = np.zeros((batchsize, outchannels, outheight, outwidth), dtype=np.float32)
        for image in prange(batchsize):
            for channel in prange(outchannels):
                for h in prange(outheight):
                    for w in prange(outwidth):
                        vert_start = h * stride
                        vert_end = vert_start + k
                        horiz_start = w * stride
                        horiz_end = horiz_start + k
        
                        subset = data[image][:,vert_start:vert_end,horiz_start:horiz_end]
                        output[image, channel, h, w] = np.sum(subset * weights[:, channel, :, :]) + bias[channel] 
        return output

    def forward(self, data):
        # TODO
        data_pad = np.pad(data, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0,0))
        output = ConvLayer.forward_numba(data_pad, self.weight.data, self.bias.data, self.stride, self.padding)
        self.data = data
        #self.fprop = output
        return output

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, dA_prev, dA_prev_pad, A_prev_pad, padding, stride, data, kernel, kernel_grad, bias_grad):
        # TODO
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        #prange, commit
        (batchsize, inchannels, height, width) = list(data.shape)
        (inchannels, outchannels, k, k) = list(kernel.shape)
        outheight = int(((height - k + 2*padding) / stride) + 1)
        outwidth = int((width - k + 2*padding) / stride + 1)

        for i in range(batchsize):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            #print(da_prev_pad.shape)
            for c in range(outchannels):
                for h in range(outheight):
                    for w in range(outwidth):
                        vert_start = h * stride
                        vert_end = vert_start + k
                        horiz_start = w * stride
                        horiz_end = horiz_start + k
                        #print(h)
                        #print(w)
                        a_slice = a_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end]
                        da_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end] += kernel[:, c, :, :] * previous_grad[i, c, h, w]
                        kernel_grad[:,c,:,:] += a_slice * previous_grad[i, c, h, w] 
                        bias_grad[c] += previous_grad[i, c, h, w]
            dA_prev[i,:,:,:] = da_prev_pad[:, padding:-padding, padding:-padding]                    
        return dA_prev

    def backward(self, previous_partial_gradient):
        # TODO
        (batchsize, inchannels, height, width) = list(self.data.shape)
        (inchannels, outchannels, k, k) = list(self.weight.data.shape)

        dA_prev = np.zeros((batchsize, inchannels, height, width), dtype=np.float32) 
        dA_prev_pad = np.pad(dA_prev, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0,0))
        A_prev_pad = np.pad(self.data, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=(0,0))
        
        dA_prev = ConvLayer.backward_numba(previous_partial_gradient, dA_prev, dA_prev_pad, A_prev_pad, self.padding, self.stride, self.data, self.weight.data, self.weight.grad, self.bias.grad)
        return dA_prev

    def selfstr(self):
        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2],
            self.weight.data.shape[3],
            self.weight.data.shape[0],
            self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(ConvLayer, self).initialize()
