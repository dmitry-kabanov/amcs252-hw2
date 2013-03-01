import numpy as np


def getMatrix(m):
    e=np.ones(m**2)
    e2=([0]+[1]*(m-1))*m
    h=1./(m+1)
    A=np.diag(-4*e,0)+np.diag(e2[1:],-1)+np.diag(e2[1:],1)+np.diag(e[m:],m)+np.diag(e[m:],-m)
    A/=h**2
    return A

def solve(m, computeRhs, bcs, plot = False, filename = ''):
    h = (1.0 - 0.0) / (m + 1.0)
    x = np.linspace(0, 1, m + 2)
    y = np.linspace(0, 1, m + 2)
    X, Y = np.meshgrid(x, y)
    matrix = getMatrix(m)
    f = computeRhs(X[1:-1, 1:-1], Y[1:-1, 1:-1], bcs, h)

    u = np.linalg.solve(matrix, f)
    u=u.reshape([m,m])
    sol = np.zeros((m+2, m+2))
    sol[1:-1, 1:-1] = u

    # Add boundary conditions values to solution.
    sol[:, 0] = bcs['left'](y)
    sol[:, -1] = bcs['right'](y)
    sol[0, :] = bcs['bottom'](x)
    sol[-1, :] = bcs['top'](x)

    # Plot solution.
    if plot:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.pcolor(X,Y,sol)
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        if filename:
            plt.savefig(filename)
        plt.show()

    return sol, X, Y
