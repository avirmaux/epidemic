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

NSTEPS = 1000 # Number of steps for ODE solver

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
