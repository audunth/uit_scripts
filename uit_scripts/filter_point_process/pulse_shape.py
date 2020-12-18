from abc import ABC, abstractmethod

import numpy as np
import warnings


class PulseShapeGenerator(ABC):
    """Abstract pulse shape, implementations should return a pulse shape vector
    given an array of time stamps with (possibly) different durations.

    For long time series this becomes expensive, and a ShortPulseShape
    implementation is preferred if the pulse function decays fast to 0
    or under machine error.
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


class StandardPulseShapeGenerator(PulseShapeGenerator):
    """Generates all pulse shapes previously supported."""

    __SHAPE_NAMES__ = {"1-exp", "lorentz", "2-exp"}
    # TODO: Implement the others

    def __init__(self, shape_name: str = "1-exp", **kwargs):
        """
        Parameters
        ----------
        shape_name Should be one of StandardPulseShapeGenerator.__SHAPE_NAMES__
        kwargs Additional arguments to be passed to special shapes:
            - "2-exp":
                - "lam" parameter for the asymmetry parameter
        """
        assert (
            shape_name in StandardPulseShapeGenerator.__SHAPE_NAMES__
        ), "Invalid shape_name"
        self._shape_name: str = shape_name
        self._kwargs = kwargs

    def get_pulse_shape(
        self, times: np.ndarray, duration: float, tolerance: float = 1e-5
    ) -> np.ndarray:

        kern = self._get_generator(self._shape_name)(times, duration, self._kwargs)
        err = max(np.abs(kern[0]), np.abs(kern[-1]))
        if err > tolerance:
            warnings.warn("Value at end point of kernel > tol, end effects may occur.")
        return kern

    @staticmethod
    def _get_generator(shape_name: str):
        if shape_name == "1-exp":
            return StandardPulseShapeGenerator._get_exponential_shape
        if shape_name == "2-exp":
            return StandardPulseShapeGenerator._get_double_exponential_shape
        if shape_name == "lorentz":
            return StandardPulseShapeGenerator._get_lorentz_shape

    @staticmethod
    def _get_exponential_shape(times: np.ndarray, duration: float, kwargs):
        kern = np.zeros(len(times))
        kern[times >= 0] = np.exp(-times[times >= 0] / duration)
        return kern

    @staticmethod
    def _get_lorentz_shape(times: np.ndarray, duration: float, kwargs):
        return (np.pi * (1 + (times / duration) ** 2)) ** (-1)

    @staticmethod
    def _get_double_exponential_shape(times: np.ndarray, duration: float, kwargs):
        lam = kwargs["lam"]
        assert (lam > 0.0) & (lam < 1.0)
        kern = np.zeros(len(times))
        kern[times < 0] = np.exp(times[times < 0] / lam / duration)
        kern[times >= 0] = np.exp(-times[times >= 0] / (1 - lam) / duration)
        return kern


class ExponentialPulseShapeGenerator(PulseShapeGenerator):
    def get_pulse_shape(self, times: np.ndarray, duration: float) -> np.ndarray:
        kern = np.zeros(times.size)
        kern[times >= 0] = np.exp(-times[times >= 0])
        return kern


class ShortPulseShapeGenerator(ABC):
    """Abstract pulse shape, implementations should return a pulse shape vector
    with (possibly) different durations.

    The length of the returned array is not restricted, this is useful
    for pulse shapes such as the exponential or the box pulse, for which
    the signal becomes zero or under machine error very quickly.

    Implementations are responsible of deciding where to place the cutoff for the returned array.
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


class ExponentialShortPulseShapeGenerator(ShortPulseShapeGenerator):
    def __init__(self, tolerance: float = 1e-50, max_cutoff: float = 1e50):
        """Exponential pulse generator, the length of the returned array is
        dynamically set to be the shortest to reach a pulse value under the
        given tolerance. That is, if the pulse shape is p(t), the returned
        array will be p(t) with t in [-T, T] such that p(-T), p(T) < tolerance.

        A max_cutoff is provided to avoid returning pulse arrays of arbitrarily long lengths.
        Parameters
        ----------
        tolerance Maximum error when cutting the pulse.
        max_cutoff
        """
        super(ExponentialShortPulseShapeGenerator, self).__init__(tolerance)
        self._max_cutoff = max_cutoff

    def get_pulse_shape(self, dt: float, duration: float):
        cutoff = -duration * np.log(self.tolerance)
        cutoff = min(cutoff, self._max_cutoff)

        tkern = np.arange(-cutoff, cutoff + dt, dt)
        kern = np.zeros(len(tkern))
        kern[tkern >= 0] = np.exp(-tkern[tkern >= 0] / duration)
        return kern


class BoxShortPulseShapeGenerator(ShortPulseShapeGenerator):
    def __init__(self, tolerance: float = 1e-50):
        super(BoxShortPulseShapeGenerator, self).__init__(tolerance)

    def get_pulse_shape(self, dt: float, duration: float):
        return np.ones(int(duration / dt))
