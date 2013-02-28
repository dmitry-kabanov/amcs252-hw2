import numpy as np
import matplotlib.pyplot as plt

import newtonmethod as nm


def G(theta, alpha, beta, h):
    g = np.zeros(len(theta))
    g[0] = theta[1] - 2 * theta[0] + alpha
    g[1:-1] = theta[:-2] - 2 * theta[1:-1] + theta[2:]
    g[-1] = theta[-2] - 2 * theta[-1] + beta
    g /= h ** 2
    g += np.sin(theta)
    return g


def J(theta, h, T):
    m = T / h - 1.
    e = np.ones(m)
    return 1. / h ** 2 * (np.diag(-2 * e, 0) + np.diag(e[:-1], -1)
                          + np.diag(e[:-1], 1)) + np.diag(np.cos(theta))


def solveBvp(m, T, t, guess, alpha, beta, filename,
             tol=2e-15, plotHistory=False, guessAndSol=False):
    h = T / (m + 1.0)
    fun = lambda (theta): G(theta, alpha, beta, h)
    jac = lambda (theta): J(theta, h, T)

    theta = np.zeros(m + 2)
    [theta[1:-1], history] = nm.solve(guess[1:-1], fun, jac, tol)
    theta[0] = alpha
    theta[-1] = beta

    if plotHistory:
        plt.figure()
        if guessAndSol:
            hh = history[0].tolist()
            hh.insert(0, alpha)
            hh.append(beta)
            plt.plot(t, hh, 'k--', label='Initial guess')
            plt.plot(t, theta, 'k-', label='Solution')
            plt.legend(loc='best')
            filename = filename + '-guess-and-sol.eps'
        else:
            for i in range(len(history)):
                hh = history[i].tolist()
                hh.insert(0, alpha)
                hh.append(beta)
                plt.plot(t, hh, label=str(i))
                plt.legend(loc='best')

            filename = filename + '-iterations.eps'
    else:
        plt.figure()
        plt.plot(t, theta, 'k-')
        filename = filename + '.eps'

    plt.xlabel(r'$t$')
    plt.ylabel(r'$\theta$')
    plt.savefig(filename)
    plt.show()
