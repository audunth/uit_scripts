def SDE(T,x0=0,a = lambda x,i: 0,b = lambda x,i: 1,seed = None):
    """
     Make a realization of dx(t) = a(x(t)) dt + b(x(t)) dW(t).
     This is the basic Runge-Kutta method for SDEs
     https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_%28SDE%29
     Usage: SDE(T,x0=0,a=lambda x,i: 0,b=lambda x,i: 1, seed = None)
     If a and b depend on some previously defined time vector V(t) (say, a = x(t) sqrt(V(t))), use a = lambda x,i: x*np.sqrt(V[i]).
     T: time array
     x0: x(t=0)
     seed: Seed for numpy random state
     Output:
         X
    """
    import numpy as np
    prng = np.random.RandomState(seed = seed)
    T = np.array(T)
    dt = sum(np.diff(T))/(len(T)-1)
    dW = prng.normal(0,dt**(1/2),len(T))
    X = np.zeros(len(T))
    X[0] = x0
    for i in range(len(T)-1):
        x = X[i]
        t = T[i]
        xi = x+a(x,i)*dt+b(x,i)*(dt**(0.5))
        X[i+1] = x + a(x,i)*dt + b(x,i)*dW[i] + 0.5*(b(xi,i)-b(x,i))*(dW[i]**2-dt)*(dt**(-0.5))
    return X
