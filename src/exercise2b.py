import numpy as np
import pendulumBvpSolver as pbs


m = 200
T = 6 * np.pi
t = np.linspace(0, T, m + 2)
alpha = 0.7
beta = 0.7

guess = 0.7 + 3.8 * np.sin(t / 6)
pbs.solveBvp(m, T, t, guess, alpha, beta, '../../images/e2b-0.7_3.8sin',
             2e-15, True, True)

guess = 0.7 + 3.9 * np.sin(t / 6)
pbs.solveBvp(m, T, t, guess, alpha, beta, '../../images/e2b-0.7_3.9sin',
             2e-5, True, True)
