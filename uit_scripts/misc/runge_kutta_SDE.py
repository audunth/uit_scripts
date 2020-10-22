from numba import jit
import numpy as np
def SDE(dt,N,x0=0,a = lambda x,i: 0,b = lambda x,i: 1,seed = None):
    """
     Make a realization of dx(t) = a(x(t)) dt + b(x(t)) dW(t).
     This is the basic Runge-Kutta method for SDEs, see e.g.
     An introduction to numerical methods for stochastic differential equations, E. Platen, Acta Numerica 8 (1999)

     Usage: SDE(dt,N,x0=0,a=lambda x,i: 0,b=lambda x,i: 1, seed = None)
     If a and b depend on some previously defined time vector V(t) (say, a = x(t) sqrt(V(t))), use a = lambda x,i: x*np.sqrt(V[i]).
     dt: time step
     N: number of iterations
     x0: x(t=0)
     seed: Seed for numpy random state
     Output:
         X
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0,sqdt,N-1)
    dW2 = 0.5*(dW**2-dt)/sqdt

    X = np.zeros(N)
    X[0] = x0
    for i in range(N-1):
        B = b(X[i],i)
        B2 = b(X[i]+B*sqdt,i) - B
        X[i+1] = X[i] + a(X[i],i)*dt + B*dW[i] + B2*dW2[i]
    return X

@jit(nopython=True)
def SDE_OU(dt, N, x0=0.,theta=1., mu=0., sigma=1., seed=None):
    """
    Fast calculation of the Ornstein-Uhlenbeck process with parameter theta, mu and sigma:
        dx_t = theta(mu-x_t)dt + sigma dW_t
    See SDE for the method (here equal to Euler-Maruyama)
    dt: time step
    N: number of iterations
    x0: initial state
    seed: seed for random state
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0,sqdt,N-1)

    X = np.zeros(N)
    X[0] = x0
    for i in range(N-1):
        X[i+1] = X[i] + theta*(mu-X[i])*dt + sigma*dW[i]
    return X

@jit(nopython=True)
def SDE_GB(dt, N, x0=0.,mu=1., sigma=1., seed=None):
    """
    Fast calculation of the geometric Brownian motion with  parameters mu and sigma:
        dx_t = mu x_t dt + sigma x_t dW_t
    See SDE for the method
    dt: time step
    N: number of iterations
    x0: initial state
    seed: seed for random state
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0,sqdt,N-1)
    dW2 = 0.5*(dW**2-dt)/sqdt

    X = np.zeros(N)
    X[0] = x0
    for i in range(N-1):
        B = sigma*X[i]
        B2 = sigma*(X[i]+B*sqdt) - B
        X[i+1] = X[i] + mu*X[i]*dt + B*dW[i] + B2*dW2[i]
    return X

@jit(nopython=True)
def SDE_SLE(dt, N, x0=0., gamma=1., log=True, seed=None):
    """
    Fast calculation of the stochastic logistic equation with parameter gamma:
        dx_t = x_t(1-x_t/(1.+gamma)) dt + sqrt(2/(1+gamma)) x_t dW_t
    See SDE for the method.
    dt: time step
    N: number of iterations
    x0: initial state
    log: If True, estimate log(x_t) and then take the exponential.
    seed: seed for random state
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0,sqdt,N-1)
    sigma = np.sqrt(2/(1.+gamma))
    X = np.zeros(N)

    if log:
        X[0] = np.log(x0)
        for i in range(N-1):
            X[i+1] = X[i] + (gamma-np.exp(X[i]))*dt/(1.+gamma) + sigma*dW[i]
        return np.exp(X)
    else:
        X[0] = x0
        dW2 = 0.5*(dW**2-dt)/sqdt
        for i in range(N-1):
            B = sigma*X[i]
            B2 = sigma*(X[i]+B*sqdt) - B
            X[i+1] = X[i] + X[i]*(1.-X[i]/(1.+gamma))*dt + B*dW[i] + B2*dW2[i]
        return X

@jit(nopython=True)
def SDE_GEXP(dt, N, x0=2., gamma=2., sqrt=True, seed=None):
    """
    Fast calculation of a gamma distributed process (shape parameter gamma, scale parameter 1) with exponential autocorrelation function:
        dx_t = (gamma-x_t) dt + sqrt(2 x_t) dW_t
    See SDE for the method.
    OBS: This is numerically stable only for gamma > 1.5 or so
        (for dt = 1e-3, the stability threshold may depend on dt)
    dt: time step normalized by correlation time
    N: number of iterations
    x0: initial state
    sqrt: If True, estimate sqrt(x_t) and then square it.
    seed: seed for random state
    """
    if seed is not None:
        np.random.seed(seed)
    sqdt = dt**0.5
    dW = np.random.normal(0,sqdt,N-1)
    X = np.zeros(N)
    if sqrt:
        X[0] = np.sqrt(x0)
        for i in range(N-1):
            X[i+1] = X[i] + (0.5/X[i])*(gamma-0.5-X[i]**2)*dt + dW[i]/np.sqrt(2.)
        return X**2
    else:
        X[0] = x0
        dW2 = 0.5*(dW**2-dt)/sqdt
        for i in range(N-1):
            B = np.sqrt(2*X[i])
            B2 = np.sqrt(2*(X[i]+B*sqdt)) - B
            X[i+1] = X[i] + (gamma-X[i])*dt + B*dW[i] + B2*dW2[i]
        return X
