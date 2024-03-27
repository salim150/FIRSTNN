import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Car_dyn:

    def __init__(self, A:torch.Tensor, B: torch.Tensor):

        self.A = A
        self.B = B

    # evaluation of the next state given current state and input
    def dynamics(self, xk:torch.Tensor, u:torch.Tensor) -> torch.Tensor:
        x_next = self.A @ xk + self.B @ u
        return x_next
