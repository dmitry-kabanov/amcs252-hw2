import numpy as np


def solve(guess, func, jacobFunc, tolerance):
    nIterMax = 100
    x = guess
    history = []
    history.append(guess)
    delta = 2 * tolerance * np.ones(len(x))
    k = 0

    while np.linalg.norm(delta, np.inf) > tolerance:
        delta = np.linalg.solve(jacobFunc(x), -func(x))
        x = x + delta
        history.append(x)
        k = k + 1

        if k > nIterMax:
            raise Exception('Newton\' method loop exceeds '
                            + str(nIterMax) + ' iterations. '
                                              'Probably, initial guess will not'
                                              ' lead to convergence.')

    return x, history