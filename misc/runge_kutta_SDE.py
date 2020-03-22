def SDE(T,x0=0,a = lambda x,i: 0,b = lambda x,i: 1,seed = None):
    """
     Make a realization of dx(t) = a(x(t)) dt + b(x(t)) dW(t).
     This is the basic Runge-Kutta method for SDEs, see e.g. 
     An introduction to numerical methods for stochastic differential equations, E. Platen, Acta Numerica 8 (1999)

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
    dW = prng.normal(0,dt**(0.5),len(T))
    X = np.zeros(len(T))
    X[0] = x0
    for i in range(len(T)-1):
        x = X[i]
        B = b(x,i)
        B2 = x+B*(dt**(0.5)) - B
        X[i+1] = x + a(x,i)*dt + B*dW[i] + 0.5*B2*(dW[i]**2-dt)*(dt**(-0.5))
    return X
