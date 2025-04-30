from abc import abstractmethod
import numpy as np
from .op import Dropout


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    # if layer.weight_decay:
                    #     layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.1):
        """
        Initialize Momentum Gradient Descent optimizer.
        
        Parameters:
        - init_lr: Initial learning rate
        - model: The neural network model
        - mu: Momentum coefficient (default: 0.9)
        """
        super().__init__(init_lr, model)
        self.mu = mu
        
        # Initialize velocity dictionary for each layer and parameter
        self.velocity = {}
        
        # Create a unique key for each layer and parameter
        for layer_idx, layer in enumerate(self.model.layers):
            if isinstance(layer, Dropout):
                continue
            if layer.optimizable:
                self.velocity[layer_idx] = {}
                for key in layer.params.keys():
                    # Initialize velocity with zeros of the same shape as the parameter
                    self.velocity[layer_idx][key] = np.zeros_like(layer.params[key])
    
    def step(self):
        """
        Perform one optimization step using momentum gradient descent.
        """
        for layer_idx, layer in enumerate(self.model.layers):
            if isinstance(layer, Dropout):
                continue
            if layer.optimizable:
                for key in layer.params.keys():
                    # Update velocity with momentum
                    self.velocity[layer_idx][key] = self.mu * self.velocity[layer_idx][key] - self.init_lr * layer.grads[key]
                    
                    # Update parameters using velocity
                    layer.params[key] += self.velocity[layer_idx][key]

                    # self.velocity[layer_idx][key] = self.mu * self.velocity[layer_idx][key] + self.init_lr * layer.grads[key]
                    
                    # # Update parameters using velocity
                    # layer.params[key] -= self.velocity[layer_idx][key]