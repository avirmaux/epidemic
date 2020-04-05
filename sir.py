""" Implementation of the SIR model

SIR model divide population into three classes: Sucpetibles, Infectious, Recovered
N people, we make the assumption that birth and death are negligeable
It is described with the set of ODE:
N * S' = - beta I S
I' = beta IS / N - gamma I
R' = gamma I

with - beta: average number of person in contact with an individual each day time probability of getting the infection
     - gamma: transition rate between I and R (if it takes D days to recover, then D = 1/gamma)

For scipy ODE solver, we than consider the ODE under the form
y' = f(t, y)
with y =[S, I, R]
"""
import math
import numpy as np
import scipy as sp
import scipy.integrate as integrate

import torch
import torch.nn as nn

NSTEPS = 1000 # Number of steps for ODE solver

def SIR_eq_batch(x, t, beta, gamma):
    """ SIR model for batch computations

    Note that it doesn't depend on t
    """
    S = x[:, 0]
    I = x[:, 1]
    R = x[:, 2]
    n = x.sum(axis=1)

    term_infection = beta * S * I / n
    term_recovery = gamma * I

    return torch.stack((S - term_infection,
                        I + term_infection - term_recovery,
                        R + term_recovery), axis=1)

class SIR:

    def __init__(self, beta, gamma):
        """ SIR model

        Args:
        -----
        beta: infection rate
        gamma: recovery rate
        """
        self.beta = beta
        self.gamma = gamma

    def simulate(self, initial_state, t_begin, t_end, n_steps):
        n = initial_state[0]
        def f(y, t):
            S, I, R = y
            return np.array([
                -self.beta * I * S / n,
                self.beta * I * S / n - self.gamma * I,
                self.gamma * I])
        steps = np.linspace(t_begin, t_end, n_steps)
        simul = integrate.odeint(f, initial_state, steps)
        return simul, steps


class SIRModelEuler(nn.Module):

    def __init__(self, step, beta=0.5, gamma=0.5, n_timesteps=10):
        """ SIR through Euler explicit method

        Args:
        -----
        n: population size
        step: Euler step
        """
        super(SIRModelEuler, self).__init__()
        # Initial parameters
        self.beta = nn.Parameter(torch.Tensor([beta]))
        self.gamma = nn.Parameter(torch.Tensor([gamma]))
        self.n_timesteps = n_timesteps
        self.step = step

    def forward(self, x):
        """ Perform explicit Euler steps from initial condition `x`
        """
        n = torch.sum(x, axis=1)
        traj = [x]
        for i in range(self.n_timesteps - 1):
            traj.append(SIR_eq_batch(x, 0, self.beta, self.gamma))

        return torch.stack(traj, axis=1)


class SIRModelRK(nn.Module):

    def __init__(self, step, beta=0.5, gamma=0.5, n_timesteps=10):
        """ SIR through RK 4 explicit method

        Args:
        -----
        n: population size
        step: Euler step
        """
        super(SIRModelRK, self).__init__()
        # Initial parameters
        self.beta = nn.Parameter(torch.Tensor([beta]))
        self.gamma = nn.Parameter(torch.Tensor([gamma]))
        self.n_timesteps = n_timesteps
        self.step = step

    def forward(self, x):
        """ Perform explicit Euler steps from initial condition `x`
        """
        n = torch.sum(x, axis=1)
        traj = [x]
        for i in range(self.n_timesteps - 1):
            # RK
            k1 = SIR_eq_batch(traj[i], 0, self.beta, self.gamma)
            k2 = SIR_eq_batch(traj[i] + self.step*k1/2, 0, self.beta, self.gamma)
            k3 = SIR_eq_batch(traj[i] + self.step*k2/2, 0, self.beta, self.gamma)
            k4 = SIR_eq_batch(traj[i] + self.step*k3, 0, self.beta, self.gamma)

            traj.append(x + (1/6) * self.step * (k1 + 2*k2 + 2*k3 + k4))

        return torch.stack(traj, axis=1)


def noise_fn(t):
    """ Yup, that's noise...
    """
    return math.sin((t+10)*(t+10))

def SIR_noise(n, n_contacts, p_contact, gamma, n_days, y0=None, noise=noise_fn):
    """ SIR model with noise

    Args:
    -----
    n: population size
    n_contacts: number of contact each day
    p_contact: probability of infction when in contact
    gamma: transition rate Infectious -> Recovered
    n_days: length of the simulation
    y0 (optional): initial state on the form [S, I, R]
    noise_fn: noise function

    beta: average number of contacts each day
    """
    BETA = n_contacts * p_contact
    def f(y, t, beta=BETA):
        # S, I, R = y[0], y[1], y[2]
        S, I, R = y
        beta += noise(t)
        beta = abs(beta) # We definitely need beta > 0
        return np.array([
            -beta * I * S / n,
            beta * I * S / n - gamma * I,
            gamma * I])

    if y0 is None:
        y0 = np.array([n, 1, 0]) # initial state: one person is sick
    sol = integrate.odeint(f, y0, np.linspace(0, n_days, NSTEPS))
    return sol

def decay_fn(beta, t):
    """ Exponential decay
    """
    return (4*beta/5) * 1/math.log(t+2) + beta/5

def SIR_decay(n, n_contacts, p_contact, gamma, n_days, y0=None, decay=decay_fn):
    """ SIR model

    Args:
    -----
    n: population size
    n_contacts: number of contact each day
    p_contact: probability of infction when in contact
    gamma: transition rate Infectious -> Recovered
    n_days: length of the simulation
    y0 (optional): initial state on the form [S, I, R]
    decay: beta decay

    beta: average number of contacts each day
    """
    BETA = n_contacts * p_contact
    def f(y, t, beta=BETA):
        # S, I, R = y[0], y[1], y[2]
        S, I, R = y
        beta = decay_fn(beta, t)
        return np.array([
            -beta * I * S / n,
            beta * I * S / n - gamma * I,
            gamma * I])

    if y0 is None:
        y0 = np.array([n, 1, 0]) # initial state: one person is sick
    sol = integrate.odeint(f, y0, np.linspace(0, n_days, NSTEPS))
    return sol
