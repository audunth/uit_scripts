from abc import ABC, abstractmethod

import numpy as np


class PulseShape(ABC):
    """
    Abstract pulse shape, implementations should return a pulse shape vector given an array of time stamps. For long
    time series this becomes expensive, and a ShortPulseShape implementation is preferred if the pulse function decays
    fast to 0 or under machine error.
    """

    @abstractmethod
    def get_pulse_shape(self, times: np.ndarray, duration: float) -> np.ndarray:
        """
        Abstract method, implementations should return np.ndarray with pulse shape with the same length as times.
        Parameters
        ----------
        times time array under which the pulse function is to be evaluated
        duration Duration time

        Returns
        -------
        np.ndarray with the pulse shape,

        """
        raise NotImplementedError


class LorentzPulseShape(PulseShape):
    def get_pulse_shape(self, times: np.ndarray, duration: float) -> np.ndarray:
        return (np.pi * (1 + times ** 2)) ** (-1)


class ExponentialPulseShape(PulseShape):
    def get_pulse_shape(self, times: np.ndarray, duration: float) -> np.ndarray:
        kern = np.zeros(times.size)
        kern[times >= 0] = np.exp(-times[times >= 0])
        return kern


class ShortPulseShape(ABC):
    """
    Abstract pulse shape, implementations should return a pulse shape vector with (possibly) different durations.
    The length of the returned array is not restricted, this is useful for pulse shapes such as the exponential or the
    box pulse, for which the signal becomes zero or under machine error very quickly.
    """

    def __init__(self, tolerance: float = 1e-50):
        self.tolerance = tolerance

    @abstractmethod
    def get_pulse_shape(self, dt: float, duration: float) -> np.ndarray:
        """
        Abstract method, implementations should return np.ndarray with pulse shape. The returned array should contain
        the values of the shape for the times [-T, T] with sampling dt, where T is such that the value of the pulse
        function is under self.tolerance. The center of the pulse should be located at the center of the array.
        Parameters
        ----------
        dt Sampling time
        duration Duration time

        Returns
        -------
        np.ndarray with the pulse shape,

        """
        raise NotImplementedError


class ExponentialShortPulseShape(ShortPulseShape):
    def __init__(self, tolerance: float = 1e-50, max_cutoff: float = 1e50):
        """
        Exponential pulse generator, the length of the returned array is dynamically set to be the shortest to reach a
        pulse value under the given tolerance. That is, if the pulse shape is p(t), the returned array will be p(t) with
        t in [-T, T] such that p(-T), p(T) < tolerance.

        A max_cutoff is provided to avoid returning pulse arrays of arbitrarily long lengths.
        Parameters
        ----------
        tolerance Maximum error when cutting the pulse.
        max_cutoff
        """
        super(ExponentialShortPulseShape, self).__init__(tolerance)
        self._max_cutoff = max_cutoff

    def get_pulse_shape(self, dt: float, duration: float):
        cutoff = -duration * np.log(self.tolerance)
        cutoff = min(cutoff, self._max_cutoff)

        tkern = np.arange(-cutoff, cutoff + dt, dt)
        kern = np.zeros(len(tkern))
        kern[tkern >= 0] = np.exp(-tkern[tkern >= 0] / duration)
        return kern


class BoxShortPulseShape(ShortPulseShape):
    def __init__(self, tolerance: float = 1e-50):
        super(BoxShortPulseShape, self).__init__(tolerance)

    def get_pulse_shape(self, dt: float, duration: float):
        return np.ones(int(duration / dt))
