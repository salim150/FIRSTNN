import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def test(nn, model, x0: torch.Tensor, T:int) -> list:
    current_state = x0
    traj, u = [], []
    traj.append(current_state)
    with torch.no_grad():
        for t in range(T):
            out = nn(torch.transpose(current_state, 0, 1))
            next_state = model.dynamics(current_state, out)
            current_state = next_state
            traj.append(current_state.detach().numpy())
            u.append(out.detach().numpy())

    return traj, u
