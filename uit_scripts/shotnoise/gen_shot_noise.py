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

def create_rate(version, gamma, K, k_length=True, tw=False):
    """Create a variable rate based on gamma using either of two methods.

    Versions:
        'ou' creates an Ornstein-Uhlenbeck process with length K or
            int(1e5), based on a SDE implemented using the library 'sdepy'.
            The processes is scaled to have a mean of gamma and returned as
            rate or as 1 / rate.
        'n-random' creates K or int(1e5) random floating pt. numbers drawn
            from a Gamma distribution (can be changed manually), where the
            scale is either gamma or 1 / gamma.

    Args:
        version (str): the version used to generate the rate array (see above)
        gamma (float): intermittency parameter
        K (int): number of arrivals
        k_length (bool, optional): If True, the rate array has length K,
                                   else, length int(1e5). Defaults to True.
        tw (bool, optional): If True, return the waiting time (i.e. 1 / rate)
                             instead of the rate process. Defaults to False.

    Returns:
        rate (np.ndarray): the rate process with shape (K,) or (int(1e5),)
        t (np.ndarray): the time axis for the rate process (same shape)
    """
    import numpy as np
    import sdepy

    T = K / gamma
    assert version in ['ou', 'n-random']
    # Use an Ornstein-Uhlenbeck process as the rate
    if version == 'ou':
        # From Matthieu Garcin. Hurst exponents and delampertized fractional Brownian motions. 2018. hal-01919754
        # https://hal.archives-ouvertes.fr/hal-01919754/document
        # and https://sdepy.readthedocs.io/en/v1.1.1/intro.html#id2
        # This uses a Wiener process as dW(t)
        @sdepy.integrate
        def rate_process(t, x, mu=2., k=.1, sigma=1.):
            return {'dt': k * (mu - x), 'dw': sigma}
        # k: speed of reversion = .1
        # mu: long-term average position = 1.
        # sigma: volatility parameter = 1.
        size = K if k_length else int(1e5)
        timeline = np.linspace(0., 1., size)
        t = np.linspace(0., T, size)
        rate = rate_process(x0=1.)(timeline)  # pylint: disable=E1102,E1123,E1120
        rate = rate.reshape((-1,))
        rate *= gamma / rate.mean()
        rate = rate if not tw else 1 / rate
    # Use n random numbers as the rate, drawn from a gamma distribution
    elif version == 'n-random':
        prob = np.random.default_rng()
        size = K if k_length else int(1e5)
        # Return rate as a waiting time if True, else as a varying gamma.
        scale = 1 / gamma if tw else gamma
        rate = prob.gamma(shape=1., scale=scale, size=size)
        # rate = prob.exponential(scale=scale, size=size)
        t = np.linspace(0, np.sum(rate), size)
    if any(rate < 0):
        print('Warning: Rate process includes negatives. Computing abs(rate).')
        rate = abs(rate)
    return rate, t

def find_nearest(array, value):
    """Find the indices of 'array' that are closest to the values in 'values'.

    Shape of array: (n,); shape of values: (k,). n>=k.

    Args:
        array (list or np.ndarray): array that is searched
        value (iterable): elements must be numerical and should
                          reflect values in 'array'

    Returns:
        idx (list): the masking of 'array' that will return an array with
                    length of 'values' and elements closest to the elements
                    of 'values'.
    """
    import numpy as np

    idx = []
    array = np.asarray(array)
    for i in value:
        idx.append((np.abs(array - i)).argmin())
    return idx

def create_ampta(rate, gamma, K):
    """Create arrival times (or waiting time scales) based on either
    of three different versions. The function 'create_rate' is called
    which generates the rate process that this function depend on.

    Versions:
        'int' uses a rate process with shape (int(1e5),) and finds the
            cumulative sum of the process. Arrival times are given as
            the indices where the cumulative sum has increased with
            some constant amount, i.e., equal spacing along the y-axis
            give the arrivals as time on the x-axis.
        '2-rd-tw' stands for double random waiting times. K random
            numbers are drawn from a Gamma distribution with scale
            =1 / gamma and shape=1. The array of K floats are returned
            and used as the scale parameters to create K waiting times,
            that is, the K waiting times are drawn from similar
            distributions but that has different scale parameter.
        'tick' uses the python library tick and the class
            SimuInhomogeneousPoisson to create arrival times from
            a rate process of length int(1e5). The library finds
            arrival times based on a varying rate (at each time step
            of the rate process the rate give the probability of an
            event / arrival) and the final number of arrivals will
            vary, but stay close to K (K must therefore be reset).

    Args:
        rate (np.ndarray): the rate process [shape=(K or int(1e5),)]
        gamma (float): intermittency parameter
        K (int): number of arrivals

    Returns:
        ta (np.ndarray): arrival times (or waiting time scales), shape
                         (K or int(1e5),).
        K (int): if 'tick', the number of arrivals is reset and returned,
                 else returns None.
    """
    import numpy as np
    import tick.base as tb
    import tick.hawkes as th
    import tick.plot as tp

    Vrate, version = rate
    T = K / gamma
    assert version in ['int', '2-rd-tw', 'tick']
    if version == 'int':
        rate, t = create_rate(Vrate, gamma, K, k_length=False)
        int_thresholds = np.linspace(0, np.sum(rate), K)
        c_sum = np.cumsum(rate)
        ta = find_nearest(c_sum, int_thresholds)
        ta = t[ta]
        # Normalize arrival times to within T_max
        ta = ta * T / np.ceil(np.max(ta))
        # amp, _, _ = gsn.amp_ta(gamma, K)
        return ta, K, T
    if version == '2-rd-tw':
        k_length = True
        tw, _ = create_rate(Vrate, gamma, K, k_length=k_length, tw=True)
        if not k_length:
            idx = np.round(np.linspace(
                0, len(tw) - 1, K)).astype(int)
            tw = tw[idx]
        # amp, ta, self.T = gsn.amp_ta(
        #     1 / tw, self.K, TWdist='gam', Adist=self.amp, TWkappa=.1)
        return tw, K, T
    if version == 'tick':
        rate, t = create_rate(Vrate, gamma, K, k_length=False)
        tf = tb.TimeFunction((t, rate))
        ipp = th.SimuInhomogeneousPoisson(
            [tf], end_time=T, verbose=False)
        ipp.track_intensity()  # Accepts float > 0 to set time step of reproduced rate process
        ipp.threshold_negative_intensity()
        ipp.simulate()
        ta = ipp.timestamps[0]
        K = len(ta)
        # amp, _, _ = gsn.amp_ta(self.gamma, self.K)
    return ta, K, T

def amp_ta(
        gamma, K, Kdist=False, mA=1., kappa=0.5, TWkappa=0.,
        TWdist='exp', Adist='exp', rate=('n-random', 'const'),
        seedTW=None, seedA=None):
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
        rate: The first position give which method to use
              when generating the variable rate process
              ('n-random' or 'ou'), and the second position
              give the method to use to generate the
              waiting times based on the given variable
              rate ('int', '2-rd-tw', 'tick'). ............. tuple of str
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
    assert rate[0] in ['n-random', 'ou'] and rate[1] in ('const', 'int', '2-rd-tw', 'tick'), 'Invalid rate tuple'

    prngTW = np.random.RandomState(seed=seedTW)
    prngA = np.random.RandomState(seed=seedA)

    if Kdist:
        K = np.random.poisson(lam=K)

    # Geneate ta, Tend
    if rate[1] in ['const', '2-rd-tw']:
        # Normal, constant waiting time (float)
        if rate[1] == 'const':
            tw = 1./gamma
        # K random waiting times (np.ndarray)
        elif rate[1] == '2-rd-tw':
            tw, _, _ = create_ampta(rate, gamma, K)
        # Generate based on either constant float or array with dim. (K,)
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
    elif rate[1] in ['int', 'tick']:
        ta, K, Tend = create_ampta(rate, gamma, K)

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


def kern(tkern, kerntype=0, lam=0.5, dkern=False, tol=1e-5):
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
        lam: Asymmetry of the two-sided exponential pulse. ... float in (0.,1.)
        dkern: Use the derivative of the kernel?  ............ bool
            Currently only implemented for kerntype=1.
        tol: Error tolerance for end effects. ................ float
    Output:
        kern: The computed kernel. ........................... (N,) np array

    All time is normalized by duration time.
    """
    import numpy as np
    import warnings
    assert(kerntype in range(5))
    kern = np.zeros(tkern.size)
    if kerntype == 0:
        kern[tkern >= 0] = np.exp(-tkern[tkern >= 0])
    elif kerntype == 1:
        assert((lam > 0.) & (lam < 1.))
        if dkern:
            kern[tkern < 0] = np.exp(tkern[tkern < 0]/lam)/lam
            kern[tkern > 0] = -np.exp(-tkern[tkern > 0]/(1.-lam))/(1.-lam)
        else:
            kern[tkern < 0] = np.exp(tkern[tkern < 0] / lam)
            kern[tkern >= 0] = np.exp(-tkern[tkern >= 0] / (1-lam))
    elif kerntype == 2:
        kern = (np.pi*(1+tkern**2))**(-1)
    elif kerntype == 3:
        kern = np.exp(-tkern**2/2)/np.sqrt(2*np.pi)
    elif kerntype == 4:
        kern = (np.pi*np.cosh(tkern))**(-1)

    err = max(np.abs(kern[0]), np.abs(kern[-1]))
    if err > tol:
        warnings.warn(
                'Value at end point of kernel > tol, end effects may occur.')

    return kern


def signal_convolve(
        A, ta, Tend, dt,
        kernsize=2**11, kerntype=0, lam=0.5, dkern=False, tol=1e-5):
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
    G = kern(tkern, kerntype, lam, dkern, tol)
    S = fftconvolve(F, G, 'same')

    return T, S


def signal_superposition(
        A, ta, Tend, dt,
        kerntype=0, lam=0.5, dkern=False):
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
        S += A[k] * kern(T-ta[k], kerntype, lam, dkern, tol=np.inf)

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
        TWdist='exp', Adist='exp', rate=('n-random', 'const'), seedTW=None, seedA=None,
        convolve=True, dynamic=False, additive=False, eps=0.1, noise_seed=None,
        kernsize=2**11, kerntype=0, lam=0.5, dkern=False, tol=1e-5):
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
            TWdist=TWdist, Adist=Adist, rate=rate, seedTW=seedTW, seedA=seedA)
    if convolve:
        T, S = signal_convolve(
                A, ta, Tend, dt,
                kernsize=kernsize, kerntype=kerntype,
                lam=lam, dkern=dkern, tol=tol)
    else:
        T, S = signal_superposition(
                A, ta, Tend, dt,
                kerntype=kerntype, lam=lam, dkern=dkern)

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
