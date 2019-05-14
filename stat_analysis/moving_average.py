def run_mean(S, radius):
    """
    Use:
        run_mean(S, radius)

    Computes the running average, using a method from
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy

    Input:
        S: Signal to be averaged. ............ (N,) np array
        radius: Window size is 2*radius+1. ... int
    Output:

    """
    import numpy as np

    window = 2*radius+1
    rm = np.cumsum(S, dtype=float)
    rm[window:] = rm[window:] - rm[:-window]
    return rm[window - 1:] / window


def run_moment(S, radius, moment=1, T=None):
    """
    Use:
        run_moment(S, radius, moment=1, T = None)

    Wrapper function for run_mean(), computes running mean and rms of S.
    To compute the running standard deviation of S, the running mean is
    subtracted from the signal.
    The running rms divides by window, not (window-1).

    Input:
        S: Signal to be averaged. ...................... (N,) np array
        radius: Window size is 2*radius+1. ............. int
        moment: Which running moment to compute. ....... in in [1,2]
                1: running mean.
                2: running standard deviation.
        T: Time base of S. ............................. (N,) np array
    Output:
        average: The running mean/rms of S, ............ (N-moment*radius,)
                 depending on moment.                    np array
        signal: Signal with values not corresponding ... (N-moment*radius,)
                to a running average removed.            np array
        time: time base corresponding to signal. ....... (N-moment*radius,)
                                                         np array
    """
    import numpy as np

    if moment == 1:
        if T is None:
            return run_mean(S, radius), S[radius:-radius]
        else:
            return run_mean(S, radius), S[radius:-radius], T[radius:-radius]
    elif moment == 2:
        rm = run_mean(S, radius)
        tmp = (S[radius:-radius]-rm)**2
        r_rms = np.sqrt(run_mean(tmp, radius))
        if T is None:
            return r_rms, S[2*radius:-2*radius]
        else:
            return r_rms, S[2*radius:-2*radius], T[2*radius:-2*radius]

# End of file moving_average.py
