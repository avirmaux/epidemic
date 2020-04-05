import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sir import SIR, SIRModelEuler, SIRModelRK

def make_dataset(beta, gamma, init_state, steps, n_timesteps):
    sir = SIR(beta=beta, gamma=gamma)
    evolution, _ = sir.simulate(init_state, steps[0], steps[-1], len(steps))
    x, y = [], []
    for i in range(len(evolution) - n_timesteps):
        x.append(evolution[i, :])
        y.append(evolution[i+1 : i+1+n_timesteps, :])
    return TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float))

BETA = 0.3
GAMMA = 0.2
N_TIMESTEMPS = 40
init_state = np.array([1, 1e-3, 0])
data = make_dataset(beta=BETA, gamma=GAMMA, init_state=init_state, steps=np.linspace(0, 40, 1000), n_timesteps=N_TIMESTEMPS)

dataset = DataLoader(data, batch_size=64, shuffle=True)
# model = SIRModelEuler(step=40/1000, n_timesteps=N_TIMESTEMPS)
model = SIRModelRK(step=40/1000, n_timesteps=N_TIMESTEMPS)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, min_lr=1e-8, patience=1000, verbose=True)
loss_fn = nn.MSELoss()
NEPOCHS = 50

for epoch in range(NEPOCHS):
    for it, (x, y) in enumerate(dataset):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
    print(f'[{epoch}] Truth: {BETA} / {GAMMA} == Model: {model.beta.item():.3f} / {model.gamma.item():.3f}')

print(BETA, GAMMA)
print(model.beta, model.gamma)
