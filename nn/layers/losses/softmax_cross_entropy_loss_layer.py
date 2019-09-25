import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(SoftmaxCrossEntropyLossLayer, self).__init__()

    def forward(self, logits, targets, axis=-1) -> float:
        # TODO
        return 0

    def backward(self):
        # TODO
        return None
