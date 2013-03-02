import sys
import numpy as np
import fivepointlaplaciannonsparse as laplaceNonSparse


def computeRhs(X, Y, bcs, h):
    """Computes right hand side."""
    f = -(20*Y**3 + 9 * np.pi**2 * (Y - Y**5)) * np.sin(3*np.pi*X)
    f[:, 0] = f[:, 0] - bcs['left'](Y[:, 0]) / h**2
    f[:, -1] = f[:, -1] - bcs['right'](Y[:, -1]) / h**2
    f[0, :] = f[0, :] - bcs['bottom'](X[0, :]) / h**2
    f[-1, :] = f[-1, :] - bcs['top'](X[-1, :]) / h**2

    return f.reshape(X.shape[0]**2)

def getLeftBc(y):
    return np.zeros(y.shape)

def getRightBc(y):
    return np.zeros(y.shape)

def getBottomBc(x):
    return np.zeros(x.shape)

def getTopBc(x):
    return np.zeros(x.shape)

m = int(sys.argv[1])
bcs = dict([
    ('left', getLeftBc),
    ('right', getRightBc),
    ('bottom', getBottomBc),
    ('top', getTopBc)])
laplaceNonSparse.solve(m, computeRhs, bcs, True)
