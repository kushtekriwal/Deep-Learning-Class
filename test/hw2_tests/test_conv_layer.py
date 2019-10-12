import pdb
import numpy as np
import torch
from torch import nn

from nn.layers.conv_layer import ConvLayer
from test import utils

TOLERANCE = 1e-4


def _test_conv_forward(input_shape, out_channels, kernel_size, stride):
    in_channels = input_shape[1]
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    original_input = input.copy()
    layer = ConvLayer(in_channels, out_channels, kernel_size, stride)

    torch_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    with torch.no_grad():
        torch_layer.weight[:] = torch.from_numpy(layer.weight.data).transpose(0, 1)
        torch_layer.bias[:] = torch.from_numpy(layer.bias.data)

    output = layer.forward(input)

    torch_data = utils.from_numpy(input)
    torch_out = utils.to_numpy(torch_layer(torch_data))
    output[np.abs(output) < 1e-4] = 0
    torch_out[np.abs(torch_out) < 1e-4] = 0

    assert np.all(input == original_input)
    assert output.shape == torch_out.shape
    assert np.allclose(output, torch_out, atol=TOLERANCE)


def test_conv_forward():
    for batch_size in range(1, 4):
        for input_channels in range(1, 4):
            for output_channels in range(1, 4):
                for width in range(10, 21):
                    for height in range(10, 21):
                        for stride in range(1, 3):
                            for kernel_size in range(stride, 4):
                                input_shape = (batch_size, input_channels, width, height)
                                _test_conv_forward(input_shape, output_channels, kernel_size, stride)


def _test_conv_backward(input_shape, out_channels, kernel_size, stride):
    in_channels = input_shape[1]
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    layer = ConvLayer(in_channels, out_channels, kernel_size, stride)

    torch_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    with torch.no_grad():
        torch_layer.weight[:] = torch.from_numpy(layer.weight.data).transpose(0, 1)
        torch_layer.bias[:] = torch.from_numpy(layer.bias.data)

    output = layer.forward(input)
    out_grad = layer.backward(np.ones_like(output))

    torch_input = utils.from_numpy(input).requires_grad_(True)
    torch_out = torch_layer(torch_input)
    torch_out.sum().backward()

    try:
        torch_out_grad = utils.to_numpy(torch_input.grad)
        out_grad[np.abs(out_grad) < 1e-4] = 0
        torch_out_grad[np.abs(torch_out_grad) < 1e-4] = 0
        assert np.allclose(out_grad, torch_out_grad, atol=TOLERANCE)

        w_grad = layer.weight.grad
        w_grad[np.abs(w_grad) < 1e-4] = 0
        torch_w_grad = utils.to_numpy(torch_layer.weight.grad.transpose(0, 1))
        torch_w_grad[np.abs(torch_w_grad) < 1e-4] = 0
        assert np.allclose(w_grad, torch_w_grad, atol=TOLERANCE)

        b_grad = layer.bias.grad
        b_grad[np.abs(b_grad) < 1e-4] = 0
        torch_b_grad = utils.to_numpy(torch_layer.bias.grad)
        torch_b_grad[np.abs(torch_b_grad) < 1e-4] = 0
        assert np.allclose(b_grad, torch_b_grad, atol=TOLERANCE)
    except:
        pdb.set_trace()
        print('bad')


def test_conv_backward():
    for batch_size in range(1, 4):
        for input_channels in range(1, 4):
            for output_channels in range(1, 4):
                for width in range(10, 21):
                    for height in range(10, 21):
                        for stride in range(1, 3):
                            for kernel_size in range(stride, 4):
                                input_shape = (batch_size, input_channels, width, height)
                                _test_conv_backward(input_shape, output_channels, kernel_size, stride)
