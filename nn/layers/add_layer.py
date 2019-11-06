from collections.abc import Iterable
from typing import Tuple

import numpy as np

from .layer import Layer


class AddLayer(Layer):
    def __init__(self, parents):
        super(AddLayer, self).__init__(parents)
        self.shape = None

    def forward(self, inputs: Iterable):
        # TODO: Add all the items in inputs. Hint, python's sum() function may be of use.
        self.shape = np.shape(inputs)
        output = sum(inputs)
        return output

    def backward(self, previous_partial_gradient) -> Tuple[np.ndarray, ...]:
        # TODO: You should return as many gradients as there were inputs.
        #   So for adding two tensors, you should return two gradient tensors corresponding to the
        #   order they were in the input.
        output = (previous_partial_gradient,) * self.shape[0]
        return output
