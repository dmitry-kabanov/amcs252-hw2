import numpy as np
import pendulumBvpSolver as pbs


m = 50
T = 2 * np.pi
t = np.linspace(0, T, m + 2)
alpha = 0.7
beta = 0.7

guess = 0.7 * np.cos(t) + 0.5 * np.sin(t)
pbs.solveBvp(m, T, t, guess, alpha, beta, '../../images/e2a-07cos+05sin',
             2e-15, True)

guess = 0.7 * np.ones(m + 2)
pbs.solveBvp(m, T, t, guess, alpha, beta, '../../images/e2a-07')

guess = 0.7 + np.sin(t / 2)
pbs.solveBvp(m, T, t, guess, alpha, beta, '../../images/e2a-07+sint_2')
