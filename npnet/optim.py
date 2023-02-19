"""
We use an optimizer to adjust the paramters of our network
based on the gradients computed during backprop
"""
from npnet.nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, lr: float = .01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad