import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def TEST(model, system, x0: torch.Tensor, T:int) -> list:
    state = x0
    u = []
    f_traj=state.detach().clone()
    with torch.no_grad():
        for t in range(T):
            out = model(torch.transpose(state, 0, 1))
            next_state = system.dynamics(state, out)
            state = next_state
            f_traj= torch.cat((f_traj,next_state.detach().clone()),1)
            u.append(out.cpu().detach().numpy())

    return f_traj, u
