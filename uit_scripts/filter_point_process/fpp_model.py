import numpy as np
from tqdm import tqdm

from uit_scripts.filter_point_process import forcing as frc, pulse_shape as ps

from typing import Callable, Tuple, Union


__COMMON_DISTRIBUTIONS__ = ["exp", "deg"]


def _get_common_distribution(
    distribution_name: str, average: float
) -> Callable[[int], np.ndarray]:
    if distribution_name == "exp":
        return lambda k: np.random.Generator.exponential(scale=average, size=k)
    elif distribution_name == "deg":
        return lambda k: average * np.ones(k)
    else:
        raise NotImplementedError


class FPPModel:
    """FPPModel is a container for all model parameters and is responsible of
    generating a realization of the process through make_realization.

    Uses a ForcingGenerator to generate the forcing, this is by default
    a StandardForcingGenerator.
    """

    def __init__(self, gamma: float, total_duration: float, dt: float):
        self.gamma = gamma
        self.T = total_duration
        self.dt = dt
        self._times: np.ndarray = np.arange(0, total_duration, dt)
        self._forcing_generator: frc.ForcingGenerator = frc.StandardForcingGenerator()
        self._pulse_generator: ps.ShortPulseGenerator = (
            ps.ExponentialShortPulseGenerator()
        )
        self._last_used_forcing: frc.Forcing = None

    def make_realization(self) -> Tuple[np.ndarray, np.ndarray]:
        result = np.zeros(len(self._times))
        forcing = self._forcing_generator.get_forcing(self._times, gamma=self.gamma)

        for k in tqdm(range(forcing.total_pulses), position=0, leave=True):
            pulse_parameters = forcing.get_pulse_parameters(k)
            self._add_pulse_to_signal(result, pulse_parameters)

        self._last_used_forcing = forcing
        return self._times, result

    def get_last_used_forcing(self) -> frc.Forcing:
        """
        Returns the latest used forcing. If several realizations of the process are run only the latest forcing will be
        available.
        -------
        """
        return self._last_used_forcing

    def set_custom_forcing_generator(self, forcing_generator: frc.ForcingGenerator):
        self._forcing_generator = forcing_generator

    def set_amplitude_distribution(
        self, amplitude_distribution: str, average_amplitude: float = 1.0
    ):
        """Sets the amplitude distribution to be used by the forcing.

        Args:
            amplitude_distribution: str
                'exp': exponential with scale parameter average_amplitude
                'deg': degenerate with location average_amplitude
            average_amplitude: float, defaults to 1.
        """
        if amplitude_distribution in __COMMON_DISTRIBUTIONS__:
            self._forcing_generator.set_amplitude_distribution(
                _get_common_distribution(amplitude_distribution, average_amplitude)
            )
        else:
            raise NotImplementedError

    def set_duration_distribution(
        self, duration_distribution: str, average_duration: float = 1.0
    ):
        """Sets the amplitude distribution to be used by the forcing.

        Args:
            duration_distribution: str
                'exp': exponential with scale parameter average_duration
                'deg': degenerate with location average_duration
            average_duration: float, defaults to 1.
        """
        if duration_distribution in __COMMON_DISTRIBUTIONS__:
            self._forcing_generator.set_amplitude_distribution(
                _get_common_distribution(duration_distribution, average_duration)
            )
        else:
            raise NotImplementedError

    def set_pulse_shape(
        self, pulse_generator: Union[ps.PulseGenerator, ps.ShortPulseGenerator]
    ):
        """
        Parameters
        ----------
        pulse_shape Instance of PulseShape, get_pulse will be called for each pulse when making a realization.
        """
        self._pulse_generator = pulse_generator

    def _add_pulse_to_signal(
        self, signal: np.ndarray, pulse_parameters: frc.PulseParameters
    ):
        """
        Adds a pulse to the provided signal array. Uses self._pulse_generator to generate the pulse shape, this can
        either be a ps.PulseGenerator or a ps.ShortPulseGenerator.
        Parameters
        ----------
        signal Signal array under construction
        pulse_parameters Parameters of the current pulse

        """
        if isinstance(self._pulse_generator, ps.PulseGenerator):
            signal += pulse_parameters.amplitude * self._pulse_generator.get_pulse(
                self._times - pulse_parameters.arrival_time,
                pulse_parameters.duration,
            )
            return

        if isinstance(self._pulse_generator, ps.ShortPulseGenerator):
            pulse = pulse_parameters.amplitude * self._pulse_generator.get_pulse(
                self.dt, pulse_parameters.duration
            )
            center_index = int(pulse_parameters.arrival_time / self.dt)
            half_length = int(len(pulse) / 2)
            from_index = max(center_index - half_length, 0)
            to_index = min(center_index + half_length, len(signal))
            pulse_from_index = from_index - (center_index - half_length)
            pulse_to_index = to_index - center_index + half_length
            signal[from_index:to_index] += pulse[pulse_from_index:pulse_to_index]
            return

        raise NotImplementedError(
            "Pulse shape has to inherit from PulseShape or ShortPulseShape"
        )
