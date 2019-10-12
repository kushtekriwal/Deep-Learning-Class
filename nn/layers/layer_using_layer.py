import pdb
from abc import ABC
from typing import List, Tuple
from nn.layers import Layer


class LayerUsingLayer(Layer, ABC):
    def __init__(self, parent=None):
        super(LayerUsingLayer, self).__init__(parent)

    @property
    def final_layer(self):
        raise NotImplementedError

    def backward(self, previous_partial_gradients=None) -> None:
        if previous_partial_gradients is not None:
            gradient = self.final_layer.backward(previous_partial_gradients)
        else:
            gradient = self.final_layer.backward()

        # Create graph
        frontier = [self.final_layer]
        graph = {}
        while len(frontier) > 0:
            node = frontier.pop()
            if node.parents is None:
                continue
            for parent in node.parents:
                if parent not in graph:
                    graph[parent] = set()
                graph[parent].add(node)
                frontier.append(parent)

        # Topological sort
        order = []
        frontier = [self.final_layer]
        while len(frontier) > 0:
            node = frontier.pop()
            order.append(node)
            if node.parents is None:
                continue
            for parent in node.parents:
                graph[parent].remove(node)
                if len(graph[parent]) == 0:
                    frontier.append(parent)

        gradients = {}
        for layer in self.final_layer.parents:
            gradients[layer] = gradient
        # Ignore loss layer because already computed
        order = order[1:]
        # Send gradients backwards
        for layer in order:
            output_grad = layer.backward(gradients[layer])
            if layer.parents is not None:
                assert isinstance(layer.parent, Tuple) == isinstance(
                    output_grad, Tuple
                ), "Gradients should be a list iff there are multiple parents."
                if not isinstance(output_grad, Tuple):
                    output_grad = (output_grad,)
                for parent, grad in zip(layer.parents, output_grad):
                    if parent in gradients:
                        gradients[parent] = gradients[parent] + grad
                    else:
                        gradients[parent] = grad
