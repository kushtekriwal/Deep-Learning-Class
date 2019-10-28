from .layer import Layer


class FlattenLayer(Layer):
    def __init__(self, parent=None):
        super(FlattenLayer, self).__init__(parent)

    def forward(self, data):
        # TODO reshape the data here and return it (this can be in place).
        shape = list(data.shape)
        return data.reshape(shape[0], -1)

    def backward(self, previous_partial_gradient):
        # TODO
        shape = list(previous_partial_gradient.shape)
        return previous_partial_gradient.reshape(shape[0], -1)
