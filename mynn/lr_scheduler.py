from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass

class Constant():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    def step(self):
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.99) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    """
    Decays the learning rate by gamma at specified milestones (epochs).
    """
    def __init__(self, optimizer, milestones=[1000, 2000, 3000, 4000], gamma=0.1) -> None:
        """
        Parameters:
        - optimizer: The optimizer whose learning rate will be adjusted
        - milestones: List of epoch indices at which to decay the learning rate
        - gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer)
        # Ensure milestones are sorted in ascending order
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.current_milestone_idx = 0
    
    def step(self) -> None:
        """
        Increment step counter and update learning rate if a milestone is reached
        """
        self.step_count += 1
        
        # Check if we've reached the next milestone
        if self.current_milestone_idx < len(self.milestones) and self.step_count >= self.milestones[self.current_milestone_idx]:
            self.optimizer.init_lr *= self.gamma
            self.current_milestone_idx += 1

class ExponentialLR(scheduler):
    """
    Decays the learning rate exponentially by gamma every epoch.
    """
    def __init__(self, optimizer, gamma=0.95) -> None:
        """
        Parameters:
        - optimizer: The optimizer whose learning rate will be adjusted
        - gamma: Multiplicative factor of learning rate decay per epoch
        """
        super().__init__(optimizer)
        self.gamma = gamma
    
    def step(self) -> None:
        """
        Increment step counter and decay learning rate exponentially
        """
        self.step_count += 1
        # Apply exponential decay every step
        self.optimizer.init_lr *= self.gamma