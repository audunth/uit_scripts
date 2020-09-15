# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:15:23 2015

@author: ath019

This file uses mpmath and numpy to calculate most common functions related to shot noise processes.

Parameters for the shot noise process:
    td: pulse duration time
    A: average pulse amplitude
    g: Intermittency parameter
    l: pulse asymmetry parameter (pulse rise time: tr = l*td, pulse fall time: tf = (1-l)*td)
        T: Total time length of signal.

"""
import numpy as np
import mpmath as mm
import warnings

###############################################################################
''' Distributions '''
###############################################################################


def shot_noise_dist(X, g, A, cdf=False):
    """
    Returns the pdf or cdf of a gamma distributed variable.
    Input:
        X: Variable values, 1d numpy array
        g: shape parameter
        A: scale parameter
        cdf: toggles pdf(default) or cdf.
    Output:
        F: The pdf or cdf of X.
    """
    F = np.zeros(len(X))
    if not cdf:
        f = lambda x, g, A: x**(g - 1) * mm.exp(-x / A) / (mm.gamma(g) * A**g)
    elif cdf:
        f = lambda x, g, A: mm.gammainc(g, a=x / A, regularized=True)
    assert(g > 0)
    assert(A > 0)
    for i in range(len(X)):
        if X[i] >= 0:
            F[i] = f(X[i], g, A)
    return F


def norm_shot_noise_dist(X, g, cdf=False):
    """
    Returns the pdf or cdf of a normalized gamma distributed variable.
    If x is gamma distributed, X=(x-<x>)/x_rms
    Input:
        X: Variable values, 1d numpy array
        g: shape parameter
        cdf: toggles pdf(default) or cdf.
    Output:
        F: The pdf or cdf of X.
    """
    F = np.zeros(len(X))
    assert(g > 0)
    if not cdf:
        f = lambda x, g: g**(g * 0.5) * (x + g**(0.5))**(g - 1) * \
            mm.exp(-g**(0.5) * x - g) / mm.gamma(g)
    elif cdf:
        f = lambda x, g: mm.gammainc(g, a=g**(0.5) * x + g, regularized=True)
    for i in range(len(X)):
        if X[i] > -g**(1 / 2):
            F[i] = f(X[i], g)
    return F


def noisy_shot_noise(X, g, e):
    """
    Returns the pdf of a normalized gamma distributed process with additive noise.
    Let z ~ Gamma(g,A), y ~ Normal(0,s^2), x = z+y.
    Input:
        X: The normalized variable X = (x-<x>)/x_rms, 1d numpy array
        g: shape parameter
        e: noise parameter, e=y_rms^2 / z_rms^2.
    Output:
        F: The pdf of X.
    """
    F = np.zeros(len(X))
    # print 'g = ', g, ', type(g) = ', type(g)
    # print 'e = ', e, ', type(e) = ', type(e)
    assert(g > 0)
    assert(e > 0)
    g = mm.mpf(g)
    e = mm.mpf(e)
    for i in range(len(X)):
        x = mm.mpf(X[i])
        # F[i] = (g/2)**(g/2)*e**(g/2-1)*(1+e)**(1/2)*mm.exp( - ((1+e)**(1/2)*x+g**(1/2))**2 / (2*e) ) *\
        #( e**(1/2)*mm.hyp1f1(g/2,1/2, ((1+e)**(1/2)*x+g**(1/2)*(1-e))**2 / (2*e) ) / (2**(1/2) * mm.gamma((1+g)/2)) +\
        #( (1+e)**(1/2)*x+g**(1/2)*(1-e) )*mm.hyp1f1((1+g)/2,3/2, ((1+e)**(1/2)*x+g**(1/2)*(1-e))**2 / (2*e) ) / mm.gamma(g/2) )

        F[i] = (g * 0.5)**(g * 0.5) * e**(g * 0.5 - 1.) * (1. + e)**(0.5) * mm.exp(-((1. + e)**(0.5) * x + g**(0.5))**(2.0) / (2.0 * e) ) *\
               (e ** (0.5) * mm.hyp1f1(0.5 * g, 0.5, ((1. + e)**(0.5) * x + g**(0.5) * (1. - e))**2 / (2. * e)) / (2.**(0.5) * mm.gamma((1. + g) * 0.5)) +
                ((1. + e)**(0.5) * x + g**(0.5) * (1. - e)) * mm.hyp1f1((1. + g) * 0.5, 1.5, ((1. + e)**(0.5) * x + g**(0.5) * (1. - e))**2 / (2. * e)) / mm.gamma(g * 0.5))
    return F


def norm_sym_dsn_dist(X, g):
    """
    Returns the normalized pdf of the derivative of a symmetric shot noise process, (td/2)*dS(t)/dt, lambda = 1/2.
    Input:
        X: The normalized variable X = (x-<x>)/x_rms, 1d numpy array
        g: shape parameter
    Output:
        F: The pdf of X.
    """
    F = np.zeros(len(X))
    assert(g > 0)
    g = mm.mpf(g)

    for i in range(len(X)):
        x = mm.mpf(np.abs(X[i]))
        F[i] = mm.sqrt(2. * g / mm.pi) * 2.**(-g / 2.) * (mm.sqrt(g) * x)**(
            (g - 1.) / 2.) * mm.besselk((1. - g) / 2., mm.sqrt(g) * x) / mm.gamma(g / 2.)

    return F


def joint_pdf_shot_noise(X, dX, g, A, l):
    # The joint PDF of X and the normalized derivative of X, dX.
    # X and dX are assumed to be 1d arrays. The returned joint PDF has
    # X on the first axis, and the returned meshgrids have 'ij'-indexing.
    # len(X) = n, len(dX) = m, shape(J) = (n,m)

    J = np.zeros([len(X), len(dX)])
    xX, dxX = np.meshgrid(X, dX, indexing='ij')
    pos = (xX + (1 - l) * dxX > 0) & (xX - l * dxX > 0)
    J[pos] = l**(g * l) * (1 - l)**(g * (1 - l)) * A**(-g) / \
        (mm.gamma(g * l) * mm.gamma(g * (1 - l)))
    J[pos] *= np.exp(-xX[pos] / A) * (xX[pos] + (1 - l) * dxX[pos]
                                      )**(g * l - 1) * (xX[pos] - l * dxX[pos])**(g * (1 - l) - 1)

    return J, xX, dxX


def shot_noise_laplace_A(X, g, a):
    """
    Returns the pdf of a shot noise process with laplace distributed amplitudes, A~Laplace(0,a)
    Input:
        X: Variable values, 1d numpy array.
        g: shape parameter
        a: scale parameter
    Output:
        F: The pdf
    """
    F = np.zeros(len(X))
    assert(g > 0)
    assert(a > 0)
    g = mm.mpf(g)
    a = mm.mpf(a)
    for i in range(len(X)):
        x = abs(X[i])
        F[i] = (x / (2 * a))**((g - 1) / 2) * mm.besselk((1 - g) /
                                                         2, x / a) / (a * np.sqrt(np.pi) * mm.gamma(g / 2))
    return F


def shot_noise_laplace_A_norm(X, g):
    """
    Returns the normalized pdf of a shot noise process with laplace distributed amplitudes, A~Laplace(0,a)
    Input:
        X: Variable values, 1d numpy array.
        g: shape parameter
    Output:
        F: The pdf
    """
    F = np.zeros(len(X))
    assert(g > 0)
    g = mm.mpf(g)
    for i in range(len(X)):
        x = abs(X[i])
        F[i] = (np.sqrt(g) * x / 2)**((g - 1) / 2) * mm.besselk((1 - g) / \
                2, np.sqrt(g) * x) * np.sqrt(g / np.pi) / mm.gamma(g / 2)
    return F

# def ALN_dist(X,a,k,e):
#    """
#    An alternative to shot_noise_laplace_A, purely based on visual comparison with the empirical PDFs.
#    Let L be an asymmetric laplace distributed variable (https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution) with scale a, asymmetry k and location m chosen m=(k^2-1)/(a k), giving <L>=0.
#    k=0 means the distirbution is a left-zero step function, k=1 gives a symmetric distribution and k->Infinity gives a right-zero step function.
#    Let N be a normally distributed variable, N~Normal(0,s). Then the ALN distribution is the distribution of X=L+N.
#    Input:
#        X: Variable values, 1d numpy array.
#        a: scale parameter
#        k: asymmetry parameter
#        e: noise parameter, e=N_rms^2 / L_rms^2
#    Output:
#        F: The PDF of X.
#    """
#    assert(a>0)
#    assert(k>0)
#    assert(e>0)
#    a=mm.mpf(a)
#    k=mm.mpf(k)
#    e=mm.mpf(e)
#    F = np.zeros(len(X))
#    # Some constants for easier computing
#    c0 = 0.5*a/(k+1/k)
#
#    c11 = e*(k**4+1)/(2*k**4) - (k**2-1)/k**2
#    c12 = -e*(k**4+1)/(k**2) + (k**2-1)
#    c13 = mm.sqrt(2*e*(k**4+1))
#
#    c21 = -e*(k**4+1)/2 + (k**2-1)
#    c22 = e*(k**4+1) + (k**2-1)
#    c23 = mm.sqrt(2*e*(k**4+1))
#
#    for i in range(len(X)):
#        x = X[i]
#        F[i] = c0 * ( mm.exp(a*x/k + c11 )*(1+mm.erf( (-a*k*x + c12)/c13 )) + mm.exp(-a*k*x + c21)*(1-mm.erf( (-a*k*x + c22)/c23 ))  )
#    return F
#
# def ALN_dist_norm(X,k,e):
#    """
#    The normalized version of ALN_dist, where a is scaled away by X->(X-<X>)/X_rms.
#    Input:
#        X: Variable values, 1d numpy array.
#        k: asymmetry parameter
#        e: noise parameter, e=N_rms^2 / L_rms^2
#    Output:
#        F: The PDF of X.
#    """
#    assert(k>0)
#    assert(e>0)
#    k=mm.mpf(k)
#    e=mm.mpf(e)
#    F = np.zeros(len(X))
#    # Some constants for easier computing
#    c0 = 0.5*mm.sqrt((1+e)*(k**4+1))/(k**2+1)
#
#    c10 = mm.sqrt((1+e)*(k**4+1))/k**2
#    c11 = e*(k**4+1)/(2*k**4) - (k**2-1)/k**2
#    c12 = -e*(k**4+1)/(k**2) + (k**2-1)
#    c13 = mm.sqrt(2*e*(k**4+1))
#    c14 = mm.sqrt((1+e)/(2*e))
#
#    c20 = -mm.sqrt((1+e)*(k**4+1))
#    c21 = -e*(k**4+1)/2 + (k**2-1)
#    c22 = e*(k**4+1) + (k**2-1)
#    c23 = mm.sqrt(2*e*(k**4+1))
#    c24 = mm.sqrt((1+e)/(2*e))
#
#    for i in range(len(X)):
#        x = X[i]
#        F[i] = c0 * ( mm.exp(c10*x + c11 )*(1+mm.erf(-c14*x + c12/c13 )) + mm.exp(c20*x + c21)*(1-mm.erf( (-c24*x + c22)/c23 ))  )
#    return F
###############################################################################
''' Autocorrelation function and power spectral density (positive half-line) '''
###############################################################################


def acorr(T, td, l):
    """
    Returns the normalized autocorrelation of a shot noise process.
    Input:
        T:  ndarray, float. Time lag.
        td: float, pulse duration time.
        l:  float, pulse asymmetry parameter. Related to pulse rise time by tr = l * td and pulse fall time by tf = (1-l) * tf.
    Output:
        R: ndarray, float. Autocorrelation at time lag tau.
    """
    R = np.zeros(T.shape[0], dtype='float64')
    assert(td > 0.0)
    assert(l >= 0.0)
    assert(l <= 1.0)

    eps = 1e-8

    td = mm.mpf(td)
    l = mm.mpf(l)
    inv_td = mm.mpf(1. / td)

    if np.abs(l) < eps or np.abs(l - 1.0) < eps:
        fun = lambda t, td, l: mm.exp(-t * inv_td)

    elif np.abs(l - 0.5) < eps:
        fun = lambda t, td, l: (1.0 + 2.0 * t * inv_td) * \
            mm.exp(-2.0 * t * inv_td)

    else:
        fun = lambda t, td, l: ((1.0 - l) * mm.exp(-t * inv_td / \
                                (1. - l)) - l * mm.exp(-t * inv_td / l)) / (1.0 - 2. * l)

    for i in range(len(T)):
        R[i] = fun(T[i], td, l)

    return R


def psd(omega, td, l):
    """
    Returns the normalized power spectral density of a shot noise process,
    given by

    PSD(omega) = 2.0 * taud / [(1 + (1 - l)^2 omega^2 taud^2) (1 + l^2 omega^2 taud^2)]

    Input:
        omega...: ndarray, float: Angular frequency
        td......: float, pulse duration time
        l.......: float, pulse asymmetry parameter.
                  Related to pulse rise time by
                  tr = l*td and pulse fall time by tf = (1-l)*tf.
    Output:
        psd.....: ndarray, float: Power spectral density
    """
    psd = np.zeros(omega.shape[0])
    assert(td > 0)
    assert(l >= 0)
    assert(l <= 1)
    #td = mm.mpf(td)
    #l = mm.mpf(l)
    if l == 0 or l == 1:
        #fun = lambda o, td, l: 4 * td / (1 + (td * o)**2)
        psd = 4. * td / (1. + (td * omega) * (td * omega))
    elif l == 0.5:
        #fun = lambda o, td, l: 64 * td / (4 + (td * o)**2)**2
        psd = 64. * td / (4. + (td * omega) * (td * omega)) ** 2.
    else:
        #fun = lambda o, td, l: 4 * td / \
        #    ((1 + ((1 - l) * td * o)**2) * (1 + (l * td * o)**2))
        psd = 4. * td / ((1. + ((1. - l) * td * omega) * (1. - l) * td * omega)
                * (1. + (l * td * omega) * (l * td * omega)))

    #for i in range(len(O)):
    #    S[i] = fun(O[i], td, l)
    return(psd)


###############################################################################
'''
Excess time statisitics

In all cases, the signal z should have been normalized as (z-<z>)/z_rms
'''
###############################################################################


def eT(X, g):
    """
    Returns the fraction of time above threshold for the normalized shot noise process X.
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
    Output:
        F: The fraction of time above threshold. The total time is T*F.
    """
    F = np.ones(len(X))
    assert(g > 0)
    g = mm.mpf(g)
    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = mm.gammainc(g, a=np.sqrt(g) * X[i] + g, regularized=True)
    return F


def eX(X, g, l):
    """
    Returns the rate of upwards level crossings above threshold for the normalized shot noise process X.
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
        l: pulse asymmetry parameter, float.
    Output:
        F: The rate of upward crossings above threshold. The total number of crossings is td*F/T.
    """
    assert(g > 0)
    assert(l >= 0)
    assert(l <= 1)
    l = mm.mpf(l)
    g = mm.mpf(g)
    F = np.zeros(len(X))

    def eXtmp(x,g,l):
        if (l>0)&(l<1):
            return ((l**(g*l-1) * (1-l)**(g*(1 - l) - 1) * g**(g / 2 - 1)
                    / (mm.gamma(g * l)* mm.gamma(g * (1 - l))))
                    * (x + np.sqrt(g))**g * mm.exp(-np.sqrt(g) * x - g))
        else:
            return (g**(g / 2)
                    * (x + np.sqrt(g))**g
                    * mm.exp(-np.sqrt(g)*x-g)
                    / mm.gamma(g))

    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = eXtmp(X[i],g,l)
    return F


def eX_l0(X, g):
    """
    Returns the rate of upwards level crossings above threshold for the normalized shot noise process X with a one sided pulse shape (l=0).
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
    Output:
        F: The rate of upward crossings above threshold. The total number of crossings is td*F/T.
    """
    warnings.warn('The functionality of eX_l0 has been added to eX.')
    assert(g > 0)
    g = mm.mpf(g)
    F = np.zeros(len(X))
    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = g**(g / 2) * (X[i] + np.sqrt(g))**g * \
                mm.exp(-np.sqrt(g) * X[i] - g) / mm.gamma(g)
    return F

# def eX_change(z,g,a):
#    # Only the function shape, not scaled. a is a free parameter.
#    # The rate of upwards crossings for a shot noise process, td*eN/T
#    F = np.zeros(len(z))
#    for i in range(len(z)):
#        if z[i]>-np.sqrt(g):
#            F[i] = a*(z[i]+np.sqrt(g))**g * mm.exp(-np.sqrt(g)*z[i]-g)
#    return F


def avT(X, g, l):
    """
    Returns the normalized average time above threshold for the normalized shot noise process X.
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
        l: pulse asymmetry parameter, float.
    Output:
        F: The normalized average time above threshold. The unnormalized version is F/td.
    """
    assert(g > 0)
    assert(l >= 0)
    assert(l <= 1)
    l = mm.mpf(l)
    g = mm.mpf(g)
    F = np.zeros(len(X))
    def avTtmp(x,g,l):
        if (l>0)&(l<1):
            return ((mm.gamma(g * l)*mm.gamma(g * (1 - l))*l**(1 - g * l)
                     * (1 - l)**(1 - g * (1 - l)) * g**(1 - g / 2) )
                     * mm.gammainc(g, a=np.sqrt(g)*x+g, regularized=True)
                     * (x + np.sqrt(g))**(-g) * mm.exp(np.sqrt(g)*x+g))
        else:
            return ((mm.gamma(g)*g**(-g/2))
                    * mm.gammainc(g, a=np.sqrt(g)*X[i]+g, regularized=True)
                    * (x+ np.sqrt(g))**(-g) * mm.exp(np.sqrt(g) * x + g))


    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = avTtmp(X[i],g,l)
    return F


def avT_l0(X, g):
    """
    Returns the normalized average time above threshold for the normalized shot noise process X with pulse asymmetry parameter l=0.
    Input:
        X: the values of the shot noise process, 1d numpy array
        g: Intermittency parameter, float
    Output:
        F: The normalized average time above threshold. The unnormalized version is F/td.
    """
    warnings.warn('The functionality of avT_l0 has been added to avT.')
    assert(g > 0)
    g = mm.mpf(g)
    F = np.zeros(len(X))
    for i in range(len(X)):
        if X[i] > -np.sqrt(g):
            F[i] = (mm.gamma(g) * g**(-g / 2)) * mm.gammainc(g, a=np.sqrt(g) * X[i] + g,
                                                             regularized=True) * (X[i] + np.sqrt(g))**(-g) * mm.exp(np.sqrt(g) * X[i] + g)
    return F


# def avT_change(z,g,a):
#    #The average time above threshold for a shot noise process, avT/td
#    # This is only the function shape, a is a free parameter.
#    F = np.zeros(len(z))
#    for i in range(len(z)):
#        if z[i]>-np.sqrt(g):
#            F[i] = a* mm.gammainc(g,a=np.sqrt(g)*z[i]+g,regularized = True) * (z[i]+np.sqrt(g))**(-g) * mm.exp(np.sqrt(g)*z[i]+g)
#    return F

# def eT_gauss(z):
#    # The fraction of time above threshold for a normally distributed process, eT/T.
#    F = np.zeros(len(z))
#    for i in range(len(z)):
#        F[i] = 0.5* mm.erfc(z[i]/np.sqrt(2))
#    return F
#
# def eX_gauss(z,Srms,dSrms):
#    # The rate of upwards crossings for a normally distributed process, td*eN/T
#    F = np.zeros(len(z))
#    for i in range(len(z)):
#        F[i] = (dSrms /(2*np.pi*Srms) )*mm.exp(-z[i]**2/2)
#    return F
#
# def avT_gauss(z,Srms,dSrms):
#    #The average time above threshold for a normally distributed process, avT/td
#    F = np.zeros(len(z))
#    for i in range(len(z)):
#        F[i] = np.pi*(Srms/dSrms)*mm.erfc(z[i]/np.sqrt(2))*mm.exp(z[i]**2/2)
#    return F

def shotnoise_PDF_laplaceA(phi_rg, gamma_val, phi_rms):
    """
    Computes the PDF for a shotnoise process with Laplace distributed Amplitudes
    A ~ Laplace(0, a)

    See O.E. Garcia and A. Theodorsen, https://arxiv.org/abs/1702.00105

    phi_rms PDF(Phi) = sqrt(gamma / pi) / Gamma(gamma / 2) * (sqrt(gamma) |Phi| / Phi_rms) ^ ((gamma - 1) / 2) * Kv((gamma-1) / 2, sqrt(gamma) |Phi| / Phi_rms)


    Input:
    ======
    phi_rg...... ndarray, float: Domain of the PDF
    gamma_val... float, intermittency parameter
    phi_rms..... float, root mean squre value of the underlying sample

    Returns:
    =======
    res......... ndarray, float: The PDF on the domain

    """

    from scipy.special import gamma as gamma_func
    from scipy.special import kv

    t1 = np.sqrt(gamma_val / np.pi) / gamma_func(0.5 * gamma_val)
    t2 = (0.5 * np.sqrt(gamma_val) * np.abs(phi_rg) /
          phi_rms) ** (0.5 * (gamma_val - 1.))
    t3 = kv(0.5 * (gamma_val - 1.), np.sqrt(gamma_val) * np.abs(phi_rg) / phi_rms)
    return (t1 * t2 * t3)


# end of file analytic_functions.py
