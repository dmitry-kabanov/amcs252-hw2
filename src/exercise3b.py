import numpy as np
import fivepointlaplaciansparse as laplaceSparse
import matplotlib.pyplot as plt

def computeRhs(X, Y, bcs, h):
    """Computes right hand side."""
    f = (2 - np.pi**2 * X**2) * np.cos(np.pi*Y)
    f[:, 0] = f[:, 0] - bcs['left'](Y[:, 0]) / h**2
    f[:, -1] = f[:, -1] - bcs['right'](Y[:, -1]) / h**2
    f[0, :] = f[0, :] - bcs['bottom'](X[0, :]) / h**2
    f[-1, :] = f[-1, :] - bcs['top'](X[-1, :]) / h**2

    return f.reshape(X.shape[0]**2)

def getLeftBc(y):
    size = y.shape
    return np.zeros(size)

def getRightBc(y):
    return np.cos(np.pi*y)

def getBottomBc(x):
    return x**2

def getTopBc(x):
    return -x**2

m = 200
bcs = dict([
    ('left', getLeftBc),
    ('right', getRightBc),
    ('bottom', getBottomBc),
    ('top', getTopBc)])
_, X, Y = laplaceSparse.solve(
    m, computeRhs, bcs, True, '../../images/e3b-solution.png')
exactSolution = X**2 * np.cos(np.pi * Y)
plt.figure()
plt.pcolor(X,Y,exactSolution)
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.savefig('../../images/e3b-exact-solution.png')
plt.show()

mList = [50, 100, 200, 500, 1000]
errorNormList = []
hList = []
for m in mList:
    print m
    numSolution, X, Y = laplaceSparse.solve(m, computeRhs, bcs)
    exactSolution = X**2 * np.cos(np.pi * Y)
    h = X[0, 1] - X[0, 0]
    error = numSolution - exactSolution
    errorNormList.append(h*np.linalg.norm(error, 2))
    hList.append(h)

plt.loglog(hList, errorNormList, 'b-o', label=r'$\|\|E\|\|_2$')
plt.loglog(hList, [0.1*h**2 for h in hList], 'g--s', label=r'$0.1h^2$')
plt.grid(True)
plt.xlabel(r'$h$')
plt.ylabel(r'$\|\|E\|\|_2$')
plt.legend(loc='best')
plt.savefig('../../images/e3b-convergence.eps')
plt.show()