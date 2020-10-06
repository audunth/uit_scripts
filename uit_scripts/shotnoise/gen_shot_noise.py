# -*- coding: utf-8 -*-
"""
Library of functions for generating shot noise processes.
It contains the following functions:

sample_asymm_laplace: generates samples for the asymmetric laplace distribution
amp_ta: generates amplitudes and arrival times for the shot noise
signal_convolve: generates the shot noise with the convolution algorithm
signal_superposition: generates the shot noise using direct superposition.
gen_noise: generates noise, either as dynamic or delta-correlated noise.
make_signal: meta-function, with all options.

In all these functions, all times are normalized by pulse duration time.
"""


def sample_asymm_laplace(
        alpha=1., kappa=0.5, size=None, seed=None):
    """
    Use:
        sample_asymm_laplace(alpha=1., kappa=0.5, size=None)
    Random samples drawn from the asymmetric Laplace distribution
    using inverse sampling. The distribution is given by
    F(A;alpha,kappa) = 1-(1-kappa)*Exp(-A/(2*alpha*(1-kappa))), A>0
                       kappa*Exp[A/(2*alpha*kappa)], A<0
    where F is the CDF of A, alpha is a scale parameter and
    kappa is the asymmetry parameter.
    Input:
        alpha: scale parameter. ......................... float, alpha>0
        kappa: shape (asymmetry) parameter .............. float, 0<=kappa<=1
        size: number of points to draw. 1 by default. ... int, size>0
        seed: specify a random seed. .................... int
    Output:
        X: Array of randomly distributed values. ........ (size,) np array
    """
    import numpy as np
    assert(alpha > 0.)
    assert((kappa >= 0.) & (kappa <= 1.))
    if size:
        assert(size > 0)
    prng = np.random.RandomState(seed=seed)
    U = prng.uniform(size=size)
    X = np.zeros(size)
    X[U > kappa] = -2*alpha*(1-kappa)*np.log((1-U[U > kappa])/(1-kappa))
    X[U < kappa] = 2*alpha*kappa*np.log(U[U < kappa]/kappa)

    return X

def sample_bounded_Pareto(alpha = 1., L = 1., H = 100., size=None, seed=None):
    """
    Use:
        sample_bounded_Pareto(alpha = 1., L = 1., H = 100., size=None, seed=None)
    random samples drawn from the bounded Pareto distribution,
        p_X(x) ~ x^(-alpha-1), L<x<H
                 0,             else
    See https://en.wikipedia.org/wiki/Pareto_distribution

    Input:
        alpha: scale parameter. ......................... float, alpha>0
        L: lower bound .................................. float, L>0
        H: upper bound .................................. float, H>L
        size: number of points to draw. 1 by default. ... int, size>0
        seed: specify a random seed. .................... int
    Output:
        X: Array of randomly distributed values. ........ (size,) np array
    """
    import numpy as np

    prng = np.random.RandomState(seed=seed)
    U = prng.uniform(size=size)

    return H*L*( (1-U)*H**alpha + U*L**alpha )**(-1./alpha)

def amp_ta(
        gamma, K, Kdist=False, mA=1., kappa=0.5, TWkappa = 0.,
        TWdist='exp', Adist='exp', seedTW=None, seedA=None):
    """
    Use:
        amp_ta(
            gamma, K, Kdist=False, mA=1., kappa=0.5, TWkappa = 0.,
            TWdist=None, Adist=None, seedTW=None, seedA=None)

    This function generates the amplitudes and arrival times
    used as the basis for a shot noise signal.
    All times are normalized by duration time.
    Input:
        gamma: average waiting time is 1/gamma. ............ float>0
        K: Number of arrivals. See Kdist, below ............ int
        Kdist: If True, K is the mean value in a Poisson distribution.
               Activate for Monte-Carlo simulations. ....... bool
        mA: mean amplitude. ................................ float>0
        kappa: asymmetry parameter for Adist. .............. float
        TWkappa: asymmetry parameter for TWdist. ........... float
        TWdist: Waiting time distribution (see below) ...... int in range(4)
        Adist: Amplitude distribution (see below) .......... int in range(5)
        seedTW/A: Specify a random seed for TWdist/Adist ... int
    Options for distributions are (where m denotes either mA or tw=1/gamma):
        Note: Using a distribution which gives negative values for the
        waiting times gives an error.
        'exp': exponential with scale parameter m
        'deg': degenerate with location m
        'ray': rayleigh with scale parameter np.sqrt(2./np.pi)*m
               (so m is the mean value)
        'unif': uniform distribution of width 2m and minimum kappa
                (kappa = 0 gives unif dist on [0,2m] and mean m)
                (kappa = -m gives unif dist on [-m,m] and mean 0)
                (for TWdist, kappa >=0.)
        'gam': gamma distribution with shape parameter kappa
               and scale parameter m/kappa (so m is the mean value)
        'alap': (only Adist) asymmetric laplace distribution with rms-value m
                and asymmetry parameter kappa.
        'pareto': (only Adist) bounded pareto distribution.
                  Lower bound 1 and upper bound mA. Scale kappa.
    Output:
        ta: Arrival times .................................. (K,) np array
        A: Amplitudes ...................................... (K,) np array
        Tend: Time length of array ......................... Float

    In order to generate the arrival times, K waiting times are drawn,
    Tw=[T_1,T_2,...,T_(K-1),T_K].
    Then the arrival times are calculated as
        ta_1 = 0
        ta_2 = T_1
        ta_3 = ta_2+T_2
        ta_4 = ta_3+T_3
        .
        .
        .
        ta_K = ta_(K-1)+T_(K-1)

    Lastly, Tend = ta_K+T_K.
    All time is normalized by duration time.
    """
    import numpy as np

    distlist = ['exp','deg','ray','unif','unifc','gam','alap', 'pareto']
    assert(TWdist in distlist[:-2]), 'Invalid TWdist'
    assert(Adist in distlist), 'Invalid Adist'

    prngTW = np.random.RandomState(seed=seedTW)
    prngA = np.random.RandomState(seed=seedA)

    if Kdist:
        K = np.random.poisson(lam=K)

    # Geneate ta, Tend
    tw = 1./gamma
    if TWdist == 'exp':
        TW = prngTW.exponential(scale=tw, size=K)
    elif TWdist == 'deg':
        TW = tw*np.ones(K)
    elif TWdist == 'ray':
        TW = prngTW.rayleigh(scale=np.sqrt(2./np.pi)*tw, size=K)
    elif TWdist == 'unif':
        assert(TWkappa>=0.), 'TWkappa>=0 for TWdist uniform'
        TW = prngTW.uniform(low=TWkappa, high=TWkappa+2*tw, size=K)
    elif TWdist == 'gam':
        TW = prngTW.gamma(TWkappa, scale=tw/TWkappa, size=K)

    TW = np.insert(TW, 0, 0.)
    ta = np.cumsum(TW[:-1])
    Tend = ta[-1] + TW[-1]

    # Generate amplitudes
    if Adist == 'exp':
        A = prngA.exponential(scale=mA, size=K)
    elif Adist == 'deg':
        A = mA*np.ones(K)
    elif Adist == 'ray':
        A = prngA.rayleigh(scale=np.sqrt(2./np.pi)*mA, size=K)
    elif Adist == 'unif':
        A = prngA.uniform(low=kappa, high=kappa+2.*mA, size=K)
    elif Adist == 'gam':
        A = prngA.gamma(kappa, scale=mA/kappa, size=K)
    elif Adist == 'alap':
        A = sample_asymm_laplace(
                alpha=mA*0.5/np.sqrt(1.-2.*kappa*(1.-kappa)), kappa=kappa,
                size=K, seed=seedA)
    elif Adist == 'pareto':
        A = sample_bounded_Pareto(alpha=kappa, L=1, H=mA, size=K, seed=seedA)

    return A, ta, Tend


def kern(tkern, kerntype=0, lam=0.5, dkern=False, tol=1e-5, shape=1, scale=1, td=1):
    """
    Use:
        kern(tkern, kerntype = 0, lam = 0.5, dkern = False, tol=1e-5)

    Returns the kernel (or pulse shape)
    Input:
        tkern: Time array for the computation. ............... (N,) np array
        kerntype: Kernel to use in the convolution: .......... int in range(4)
            0: one-sided exponential pulse
            1: two-sided exponential pulse (requires lam in (0,1)).
            2: Lorenz pulse
            3: Gaussian pulse
            4: Sech pulse
            5: Box pulse
            6: Triangular pulse
            7: Rayleigh pulse
            8: Gamma pulse
            9: Pareto pulse
            10: Laplace pulse
        lam: Asymmetry of the two-sided exponential pulse. ... float in (0.,1.)
        dkern: Use the derivative of the kernel?  ............ bool
            Currently only implemented for kerntype=1.
        tol: Error tolerance for end effects. ................ float
        shape: Shape parameter for Gamma and Pareto pulse. ... float > 0
        scale: scale parameter where needed. ................. float > 0
    Output:
        kern: The computed kernel. ........................... (N,) np array

    All time is normalized by duration time.
    """
    import numpy as np
    import warnings
    assert(kerntype in range(11))
    assert(scale > 0.)
    assert(shape > 0.) 
    kern = np.zeros(tkern.size)
    if kerntype == 0:
        kern[tkern >= 0] = 1/scale*np.exp(-tkern[tkern >= 0]/(scale*td))
    elif kerntype == 1:
        assert((lam > 0.) & (lam < 1.))
        if dkern:
            # duration time not implemented for derivative yet
            kern[tkern < 0] = np.exp(tkern[tkern < 0]/lam)/lam
            kern[tkern > 0] = -np.exp(-tkern[tkern > 0]/(1.-lam))/(1.-lam)
        else:
            kern[tkern < 0] = 1/scale*np.exp(tkern[tkern < 0] / lam/(scale*td))
            kern[tkern >= 0] = 1/scale*np.exp(-tkern[tkern >= 0] / (1-lam)/(scale*td))
    elif kerntype == 2:
        kern = (scale*np.pi*(1+(tkern/(scale*td))**2))**(-1)
    elif kerntype == 3:
        kern = np.exp(-(tkern/(scale*td))**2/2)/(np.sqrt(2*np.pi)*scale)
    elif kerntype == 4:
        kern = (np.pi*np.cosh(tkern/td))**(-1)
    elif kerntype == 5:
        kern[tkern >-0.5*td] = 1
        kern[tkern > 0.5*td] = 0
    elif kerntype == 6:
        kern[tkern >= -1*td] = 1 + tkern[tkern >= -1*td]/td
        kern[tkern >= 0*td] = 1 - tkern[tkern >= 0*td]/td
        kern[tkern >= 1*td] = 0
    elif kerntype == 7:
        kern[tkern >= 0] = tkern[tkern >= 0]/td/scale**2 * np.exp( - (tkern[tkern >= 0]/td)**2/(2*scale**2))
    elif kerntype == 8:
        from scipy.special import gamma as Ga
        kern[tkern >= 0] = 1/(Ga(shape)*scale**shape) * (tkern[tkern >= 0]/td)**(shape-1) * np.exp( - (tkern[tkern >= 0]/td)/scale)
    elif kerntype == 9:
        # add scale to tkern -> peak of pulse is at t=0
        kern[tkern >= 0] = shape*scale**shape/(tkern[tkern >= 0]/td+scale)**(shape+1)
    elif kerntype == 10:
        kern = 1/(2*scale) * np.exp(-np.abs(tkern/td)/scale)


    err = max(np.abs(kern[0]), np.abs(kern[-1]))
    if err > tol:
        warnings.warn(
                'Value at end point of kernel > tol, end effects may occur.')

    return kern


def signal_convolve(
        A, ta, Tend, dt,
        kernsize=2**11, kerntype=0, lam=0.5, dkern=False, tol=1e-5, kernshape=1, kernscale=1):
    """
    Use:
        signal_convolve(
            A, ta, Tend, dt,
            kernsize=2**11, kerntype=0, lam=0.5, dkern=False, tol=1e-5)

    The shot noise process can be calculated as S(t) = [G*F](t)
    where * denotes a convolution, G is a kernel function and
    F is a train of K delta pulses arriving at times [ta_1,...,ta_K]
    with amplitudes [A_1,...,A_K].

    The arrival times are approximated to time indexes,
    thus there is some error in the pulse locations.
    Therefore, dt<= 0.01 is reccomended.
    Input:
        A: Amplitude array. ................................... (K,) np array
        ta: Arrival times of pulses. .......................... (K,) np array
        Tend: Time length of signal. .......................... float
        dt: Time step. Should be 10^(-2) or smaller. .......... float
        kernsize: the kernel is 2*kernsize+1 data points. ..... int
        kerntype, lam, dkern, tol: See the kern() function.
    Output:
        T: time array with N = ceil(Tend/dt).astype(int)+1. ... (N,) np array
        S: shot noise signal of lenght len(T). ................ (N,) np array

    All time is normalized by duration time.
    """
    import numpy as np
    from scipy.signal import fftconvolve

    T = np.arange(0, np.ceil(Tend/dt).astype(int)+1)*dt
    tkern = np.arange(-kernsize, kernsize+1)*dt

    def genF():
        # Make a K-sized array of arrival time indexes
        ta_index = np.ceil(ta/dt).astype(int)
        F = np.zeros(T.size)
        for i in range(ta_index.size):
            F[ta_index[i]] += A[i]  # This may need to be multiplied by td.
        return F

    F = genF()
    G = kern(tkern, kerntype, lam, dkern, tol, shape=kernshape, scale=kernscale)
    S = fftconvolve(F, G, 'same')

    return T, S


def signal_superposition(
        A, ta, Tend, dt,
        kerntype=0, lam=0.5, dkern=False, kernshape=1, kernscale=1):
    """
    Use:
        signal_superposition(
            A, ta, Tend, dt,
            kerntype=0, lam=0.5, dkern=False)

    Calculates a shot noise process as a superposition of pulses.
    Input:
        A: Amplitude array. ................................... (K,) np array
        ta: Arrival times of pulses. .......................... (K,) np array
        Tend: Time length of signal. .......................... float
        dt: Time step. ........................................ float
        kerntype, lam, dkern: See the kern() function.
    Output:
        T: time array with N = ceil(Tend/dt).astype(int)+1. ... (N,) np array
        S: shot noise signal of lenght len(T). ................ (N,) np array

    All time is normalized by duration time.
    """
    import numpy as np

    T = np.arange(0, np.ceil(Tend/dt).astype(int)+1)*dt

    S = np.zeros(T.size)
    K = A.size

    for k in range(K):
        # tol in kernel set to np.inf as the kernel is computed
        # over the entire time array. End effects do not cause problems.
        S += A[k] * kern(T-ta[k], kerntype, lam, dkern, tol=np.inf, shape=kernshape, scale=kernscale)

    return T, S


def gen_noise(
        gamma, eps, T, mA=1.,
        kernsize=2**11, kerntype=0, lam=0.5, dkern=False, tol=1e-5,
        noise_seed=None):
    """
    Use:
        gen_noise(
            gamma, eps, T, mA=1.,
            kernsize=2**11, kerntype=0, lam=0.5, dkern=False, tol=1e-5,
            noise_seed=None)
    Calculates zero-mean normally distributed noise for the shot noise process.
    Input:
        gamma: Intermittency parameter. ............. float
        eps: Noise to signal ratio. ................. float
            defined as X_rms/S_rms where X is noise and S is signal.
        T: time array of size N. .................... (N,) np array
        mA: Mean pulse amplitude. ..... float
        kernsize: size of kernel is 2*kernsize+1. ... int
        kerntype, lam, dkern, tol: See the kern() function.
        noise_seed: Random seed. .................... int
    Output:
        res: The computation result. ................ 2 (N,) np array
            A 2-tuple consisting of [dynamic noise, additive noise]


    All time is normalized by duration time.
    """
    import numpy as np
    from scipy.signal import fftconvolve
    prng = np.random.RandomState(seed=noise_seed)
    sigma = np.sqrt(eps*gamma)*mA

    N = T.size
    dt = np.diff(T)[0]

    res = [None, None]

    tkern = np.arange(-kernsize, kernsize+1)*dt
    G = kern(tkern, kerntype, lam, dkern, tol)
    dW = prng.normal(scale=np.sqrt(2*dt), size=N)
    res[0] = sigma*fftconvolve(dW, G, 'same')

    res[1] = sigma*prng.normal(size=N)

    return res


def make_signal(
        gamma, K, dt, Kdist=False, mA=1., kappa=0.5, TWkappa=0, ampta=False,
        TWdist='exp', Adist='exp', seedTW=None, seedA=None, convolve=True,
        dynamic=False, additive=False, eps=0.1, noise_seed=None,
        kernsize=2**11, kerntype=0, lam=0.5, dkern=False, tol=1e-5, kernshape=1, kernscale=1):
    """
    Use:
        make_signal(
            gamma, K, dt, Kdist=False, mA=1.,kappa=0.5, TWkappa=0, ampta=False,
            TWdist='exp', Adist='exp', seedTW=None, seedA=None, convolve=True,
            dynamic=False, additive=False, eps=0.1, noise_seed=None,
            kernsize=2**11, kerntype=0, lam=0.5, dkern=False)

    Meta-function with all options. Calls all the above functions.
    Input:
        See the other functions for explanation.
        amptd: If True, returns amplitudes and duration times as well.
    Output:
        The output is given in the following order:
        T, S, S+dynamic noise, S+additive noise, A, ta
        Only outputs noise or amplitudes and duration times if prompted.

    All time is normalized by duration time.
    """
    import numpy as np
    A, ta, Tend = amp_ta(
            gamma, K, Kdist=Kdist, mA=mA, kappa=kappa, TWkappa = TWkappa,
            TWdist=TWdist, Adist=Adist, seedTW=seedTW, seedA=seedA)
    if convolve:
        T, S = signal_convolve(
                A, ta, Tend, dt,
                kernsize=kernsize, kerntype=kerntype,
                lam=lam, dkern=dkern, tol=tol, kernshape=kernshape, kernscale=kernscale)
    else:
        T, S = signal_superposition(
                A, ta, Tend, dt,
                kerntype=kerntype, lam=lam, dkern=dkern, kernshape=kernshape, kernscale=kernscale)

    if (dynamic or additive):
        X = gen_noise(
                gamma, eps, T, mA=mA,
                kernsize=kernsize, kerntype=kerntype,
                lam=lam, dkern=dkern, tol=tol,
                noise_seed=noise_seed)

    res = (T,S)
    if dynamic:
        res += (S+X[0],)
    if additive:
        res += (S+X[1],)
    if ampta:
        res += (A,ta)
    return res

# End of file shot_noise_functions.py
