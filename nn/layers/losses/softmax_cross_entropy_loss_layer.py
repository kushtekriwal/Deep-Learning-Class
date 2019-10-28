import numpy as np
import pandas as pd

from .loss_layer import LossLayer
from sklearn.preprocessing import OneHotEncoder

class SoftmaxCrossEntropyLossLayer(LossLayer):
    def __init__(self, reduction="mean", parent=None):
        """

        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)
        self.reduction = reduction  
        self.axis = None
        self.target = None
        self.exp = None
        self.size = None

    def forward(self, logits, targets, axis=-1) -> float:
        """

        :param logits: ND non-softmaxed outputs. All dimensions (after removing the "axis" dimension) should have the same length as targets
        :param targets: (N-1)D class id integers.
        :param axis: Dimension over which to run the Softmax and compare labels.
        :return: single float of the loss.
        """
        # TODO
        #shape = list(logits.shape)
        max_logits = np.amax(logits, axis=axis)
        logits2 = logits - np.expand_dims(max_logits, axis=axis) 
        self.exp = np.exp(logits2)
        log_softmax = logits2 - np.expand_dims(np.log(np.sum(np.exp(logits2), axis=axis)), axis=axis)


        numcols = np.size(logits, axis)
        numvals = targets.size
        self.size = numvals
        onehot = np.zeros((numvals, numcols), dtype=np.float32)
        batchsize = np.arange(numvals)
        onehot[batchsize, targets.flatten()] = 1.0

        log_softmax_move = np.moveaxis(log_softmax, axis, -1)
        shape2 = list(log_softmax_move.shape)
        cross_entropy = -1 * np.sum(log_softmax_move.reshape(-1, shape2[-1]) * onehot)
        if self.reduction == "mean":
            cross_entropy = cross_entropy / numvals
        self.logits = logits
        self.target = onehot
        self.axis = axis
        return cross_entropy

    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        # TODO
        shape = list(self.logits.shape)
        softmax = self.exp / np.expand_dims(np.sum(self.exp, axis=self.axis), axis=self.axis)
        softmax_move = np.moveaxis(softmax, self.axis, -1)
        shape2 = list(softmax_move.shape)
        res = softmax_move.reshape(-1, shape2[-1]) - self.target 
        res = res.reshape(shape2)
        res = np.moveaxis(res, -1, self.axis)
        if self.reduction == "mean":
            res = res / self.size
        return res
