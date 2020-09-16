# Note that the python package fbm can also be used. 

def DH_gen_fGn(length, hurst):
    """
    Usage:
    fgn = DH_gen_fGn(length, hurst)
    Uses the Davies-Harte method [1] for simulating a fractional Gaussian noise
    process. The method actually used is detailed in [2].

    Small length combined with large hurst can give negative fourier
    transformed coefficients and will raise an error.

    Even if this does not happen,
    You should check that the standard deviation is close to 1, as too small
    time series length is bad for the simulation.



    Input:
        length: time series length. .................. int
        hurst: Hurst parameter in (0,1). ............. float, 0<hurst<1

    Output:
        fgn: realization of fGn. ..................... 1D np array

    Refs:
    [1]: Davies, R. B and Harte, D. S., Biometrika 74, 1, pp. 95-101 (1987)
    [2]: Dieker, A., 'Simulation of fractional Brownian motion'. PhD thesis,
         University of Amsterdam (2006)
         http://www.columbia.edu/~ad3217/fbm/thesis.pdf
    """
    import numpy as np

    assert((hurst > 0) & (hurst < 1)), 'hurst not in (0,1).'
    if hurst == 0.5:
        return np.random.normal(0., 1., length)
    else:
        def gen_c():
            def rho(i):
                return (0.5*np.abs(i+1.)**(2.*hurst)
                        + 0.5*np.abs(i-1.)**(2.*hurst)
                        - np.abs(i)**(2.*hurst))

            def ac(t, s):
                return 1.*rho(t-s)

            ctmp = ac(np.arange(1, length), 0)
            c = np.zeros(2*length)
            c[0] = ac(0., 0.)
            c[1:length] = ctmp[:]
            c[length] = 0
            c[length+1:] = ctmp[::-1]
            return c

        g = np.fft.fft(gen_c())
        assert(g.real[g.real < 0].size == 0), 'negative eigenvalues'

        def gen_Z():
            Z = np.zeros(2*length, dtype=complex)
            Z[0] = np.random.normal(0., 1.)
            Z[length] = np.random.normal(0., 1.)
            V = np.random.normal(0., 1., (length-1, 2))
            Z[1:length] = (V[:, 0]+1.j*V[:, 1])/np.sqrt(2)
            Z[length+1:] = np.conj(Z[1:length][::-1])
            return Z

        fgn = np.sqrt(2*length)*np.fft.ifft(np.sqrt(g)*gen_Z()).real[:length]
        print('fgn realization: mean={0:.3f}, std={1:.3f}'.format(fgn.mean(),
                                                                  fgn.std()))
        return fgn


# EOF gen_fGn.py
