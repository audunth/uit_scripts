# Methods for performing Richardson-Lucy deconvolution.
# A kernel (pulse shape) is needed.
#
# First, use RL_gauss_deconvolve to perform the deconvolution.
# More iterations are better.
# Check that the result falls to zero during quiet periods, or amplitudes
# may not be correctly calculated.
#
# Then, use find_amp_ta to calculate the peaks. The default values work OK.
# Noise is handeled by increasing window_length or order in find_amp_ta.


def RL_gauss_deconvolve(sig, kern, iterlist,
                        init=None, shift = None, cutoff=1e-10):
    """
    Use: RL_gauss_deconvolve(sig,kern, iterlist, init=None, cutoff=1e-10)
    Performs the Richardson-Lucy deconvolution for normally distributed noise.
    See https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    and https://arxiv.org/abs/1802.05052.

    Input:
        sig: signal to be deconvolved ............................. 1D np array
        kern: deconvolution kernel ................................ 1D np array
        iterlist: the number of iterations. ....................... int or
              If this is a list, the deconvolution result           list of int
              is returned for each element in iterlist, see below.
        init: initial array guess. Leave blank for all zeros. ..... 1D np array
        shift: shift parameter for removing negative values. ...... float
        cutoff: for avoiding divide by zero errors. ............... float

    Output:
        res: result array. NxM, where N=len(sig) and M=len(iterlist)   np array
        err: mean absolute difference between iterations .......... 1D np array
    """
    import numpy as np
    from scipy.signal import fftconvolve

    if init is None:
        update0 = np.ones(sig.size)
        update1 = np.ones(sig.size)
    else:
        update0 = np.copy(init)
        update1 = np.copy(init)

    sigtmp = np.copy(sig)
    kerntmp = np.copy(kern)
    if shift:
        alpha = np.sum(kern)/np.sum(sig)
        sigtmp += shift
        kerntmp += shift*alpha

    if type(iterlist) is int:
        iterlist = [iterlist, ]

    err = np.zeros(iterlist[-1]+1)
    err[0] = np.sum((sigtmp-fftconvolve(update0,kerntmp,'same'))**2)
    res = np.zeros([sig.size, len(iterlist)])

    kern_inv = kerntmp[::-1]
    sigconv = fftconvolve(sigtmp, kern_inv, 'same')
    kernconv = fftconvolve(kerntmp, kern_inv, 'same')

    index_array = np.arange(sigtmp.size)
    count = 0
    for i in range(1, iterlist[-1]+1):
        # If an element in the previous iteration is very close to zero,
        # the same element in the next iteration should be as well.
        # This is handeled numerically by setting all elements <= cutoff to 0
        # and only performing the interation on those elements > cutoff.

        tmp = fftconvolve(update0, kernconv, 'same')
        good = tmp > cutoff

        #update1 = update0 * sigconv / tmp
        update1[good] = update0[good] * sigconv[good] / tmp[good]

        #bad = index_array[np.invert(good)]
        #if len(bad)>0:
        #    print('iter {}, bad idx {}, bad update vals {}'.format(i,bad,update1[bad]))

        #err[i-1] = np.mean(np.abs(update1-update0))
        err[i] = np.sum((sigtmp-fftconvolve(update1,kerntmp,'same'))**2)
        update0[:] = update1[:]

        if i == iterlist[count]:
            print('i = {}'.format(iterlist[count]), flush=True)
            res[:, count] = update1[:]
            count += 1

    return res, err

def find_amp_ta_savgol(D, T, window_length=3):
    """
    This tests a new method of finding minima between peaks.

    Use: ta,amp = find_amp_ta(D, T, window_length=3)
    Estimates arrival times and amplitudes
    of the FPP from the deconvolved signal D.

    Input:
        D: result of deconvolution ............... numpy array
        T: time array ............................ numpy array
        dt: time step ............................ float
        window_length: passed to savgol_filter ... int >= 3, default 3

    Output:
        ta: estimated arrival times .............. numpy array
        amp: estimated amplitudes ................ numpy array


    To find ta, the derivative of D is computed
    using a Savitzky-Golay of polynomial order 2.
    The peaks are found from the zero-crossings of this derivative
    in the negative direction.
    Polynomial order 2 is chosen as it seems best for single peaks.
    In the presence of noise, increasing window_length increases smoothing.

    In order to take the entire mass of each peak of D into account,
    the amplitudes are estimated by summing from one minima between two peaks
    to the minima between the next two peaks, as determined by the positive
    zero-crossings of the derivative of D.

    ---min---peak----min----peak----min---

    ---][--sum range-][--sum range--][----
    """
    import numpy as np
    from scipy.signal import savgol_filter

    dt = np.mean(np.diff(T))

    # Find indices of arrival times
    polyorder = 2
    dD = savgol_filter(D, window_length, polyorder, deriv=1, delta=dt)

    places = np.where(dD >= 0)[0]
    dplaces = places[1:]-places[:-1]
    split = np.where(dplaces != 1)[0]+1
    lT = np.split(places, split)
    peak = np.zeros(len(lT), dtype=int)
    for i in range(len(lT)):
        peak[i] = lT[i][-1]

    # Find minima between arrivals
    places = np.where(dD < 0)[0]
    dplaces = places[1:]-places[:-1]
    split = np.where(dplaces != 1)[0]+1
    lT = np.split(places, split)
    interpeak = np.zeros(len(lT), dtype=int)
    for i in range(len(lT)):
        interpeak[i] = lT[i][-1]

    # Control that interpeaks surround peaks:
    if interpeak[0]>peak[0]:
        interpeak = np.insert(interpeak,0,0)
    if peak[-1]>interpeak[-1]:
        interpeak = np.append(interpeak,D.size-1)
    assert(interpeak.size == peak.size+1)

    # Find amplitudes
    amp = np.zeros(peak.size)
    for i in range(amp.size):
        amp[i] = np.sum(D[interpeak[i]:interpeak[i+1]])
    if interpeak[-1]<D.size:
        amp[-1] = np.sum(D[interpeak[i]:interpeak[i+1]])
    else:
        amp[-1] = np.sum(D[interpeak[i]:])
    # Often, zero-mass peaks are found. Remove these.
    ta = T[peak][amp>0]
    amp = amp[amp>0]

    return ta, amp


def find_amp_ta_old(D, T, savgol=True, window_length=3, order=1):
    """
    Use: ta,amp = find_amp_ta(D, T, savgol=True, window_length=3, order=1)
    Estimates arrival times and amplitudes
    of the FPP from the deconvolved signal D.

    Input:
        D: result of deconvolution ............... numpy array
        T: time array ............................ numpy array
        dt: time step ............................ float
        savgol: if True, use the savgol filter.
                if False, use argrelmax. ......... bool, default True
        window_length: passed to savgol_filter ... int >= 3, default 3
        order: passed to argrelmax.
               only used for savgol=False. ....... int >= 1, default 1

    Output:
        ta: estimated arrival times .............. numpy array
        amp: estimated amplitudes ................ numpy array


    To find ta, one of two methods is used.

    If savgol=True, the derivative of D is computed
    using a Savitzky-Golay of polynomial order 2.
    The peaks are found from the zero-crossings of this derivative
    in the negative direction.
    Polynomial order 2 is chosen as it seems best for single peaks.
    In the presence of noise, increasing window_length increases smoothing.

    If savgol = False, the peaks are found using scipy.signal.argrelmax.
    In this case, order is passed to argrelmax.

    In order to take the entire mass of each peak of D into account,
    the amplitudes are estimated by summing from one minima between two peaks
    to the minima between the next two peaks:

    ---min---peak----min----peak----min---

    ---][--sum range-][--sum range--][----
    """
    import numpy as np
    from scipy.signal import argrelmax, savgol_filter

    dt = np.mean(np.diff(T))
    if savgol:
        # Find indices of arrival times
        polyorder = 2
        dD = savgol_filter(D, window_length, polyorder, deriv=1, delta=dt)

        places = np.where(dD > 0)[0]
        dplaces = places[1:]-places[:-1]
        split = np.where(dplaces != 1)[0]+1
        lT = np.split(places, split)
        peak = np.zeros(len(lT), dtype=int)
        for i in range(len(lT)):
            peak[i] = lT[i][-1]
    else:
        peak = argrelmax(D, order=order)[0]

    # Find amplitudes
    amp = np.zeros(peak.size)
    interpeak = np.zeros(peak.size+1, dtype=int)
    for i in range(peak.size-1):
        interpeak[i+1] = peak[i]+np.argmin(D[peak[i]+1:peak[i+1]])
    interpeak[-1] = D.size
    for i in range(amp.size):
        amp[i] = np.sum(D[interpeak[i]:interpeak[i+1]])

    return T[peak], amp

def find_amp_ta_test(D, T, window_length=0, order=1):
    """
    Use: ta,amp = find_amp_ta_test(D, T, window_length=0, order=1)
    Estimates arrival times and amplitudes of the FPP
    from the deconvolved signal D.

    Input:
        D: result of deconvolution ............... numpy array
        T: time array ............................ numpy array
        dt: time step ............................ float
        window_length: passed to savgol_filter.
                        not used by default. ..... odd int,
                                                    default 0 (no filtering)
        order: passed to argrelmax. There should
                be no need for modifying this. ... int >= 1, default 1

    Output:
        ta: estimated arrival times .............. numpy array
        amp: estimated amplitudes ................ numpy array

    A slightly different method for finding amplitudes and arrivals.
    The peaks are estimated using a standard scipy.signal.argrelmax,
    but we apply a smoothing by using a Savitzky-Golay filter of
    polynomial order 2 before argrelmax is applied.
    Polynomial order 2 is chosen as it seems best for single, positive definite
    peaks. Anything higher gives spurious fluctuations.
    In the presence of noise, increasing window_length increases smoothing.

    In order to take the entire mass of each peak of D into account,
    the amplitudes are estimated by summing from one minima between two peaks
    to the minima between the next two peaks:

    ---min---peak----min----peak----min---

    ---][--sum range-][--sum range--][----
    """
    import numpy as np
    from scipy.signal import argrelmax, savgol_filter
    import warnings
    # Pad D with zeros at ends in case there is a peak at the end points.
    # This is neccesary as argrelmax does not take end points into account.
    Dpad = np.zeros(D.size+2*order)
    Dpad[order:-order] = D[:]

    try:
        Dpad = savgol_filter(Dpad,window_length,polyorder=2,mode='constant')
    except ValueError:
        if window_length:
            warnings.warn('No filtering used. Check window_length.',UserWarning)

    # Peaks are detected on the (possibly) filtered Dpad
    peak = argrelmax(Dpad,order=order)[0]

    amp = np.ones(peak.size)
    interpeak = np.zeros(peak.size+1, dtype=int)
    for i in range(peak.size-1):
        interpeak[i+1] = peak[i]+1+np.argmin(Dpad[peak[i]+1:peak[i+1]])
    interpeak[-1] = Dpad.size-1

    # As the filtering preserves the integral of the time series,
    # the amplitudes are also calculated on the filtered Dpad.
    for i in range(amp.size):
        amp[i] = np.sum(Dpad[interpeak[i]:interpeak[i+1]])

    return T[peak-order],amp



# EOF deconv_methods
