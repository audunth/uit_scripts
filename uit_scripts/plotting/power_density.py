import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal


def lineal(x, a, s):
    return a * x + s


def plot_pds(x, dt, cutoff_min, cutoff_max):
    fig, ax = plt.subplots(1, 2)
    ax[0].axvline(cutoff_min, ls="dashed", alpha=0.5)
    ax[0].axvline(cutoff_max, ls="dashed", alpha=0.5)

    f, Pxx_den = signal.welch(x, fs=1 / dt, nperseg=2 ** 20)
    ax[0].scatter(f, Pxx_den)
    popt, pcov = curve_fit(
        lineal,
        np.log10(f[np.logical_and(f > cutoff_min, f < cutoff_max)]),
        np.log10(Pxx_den[np.logical_and(f > cutoff_min, f < cutoff_max)]),
        maxfev=10000,
    )
    ax[0].plot(f, 10 ** lineal(np.log10(f), *popt), "r-")

    ax[0].set_ylabel(r"$\log(\Phi(f))$")
    ax[0].set_xlabel(r"$\log(f)$")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_title(r"$\Phi(f)$")
    ax[0].set_xlim(cutoff_min / 1e2, cutoff_max * 1e2)
    ax[0].text(
        0.6,
        0.7,
        r"$\beta \approx {:.2f}({:.2f})$".format(-popt[0], np.sqrt(pcov[0, 0])),
        transform=ax[0].transAxes,
    )

    ax[1].axvline(cutoff_min, ls="dashed", alpha=0.5)
    ax[1].axvline(cutoff_max, ls="dashed", alpha=0.5)
    ax[1].scatter(f, np.power(f, -popt[0]) * Pxx_den)
    ax[1].set_title(r"$\Phi(f) f ^ {\beta}$")  # .format(-popt[0]))
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\log(f)$")
    ax[1].set_xlim(1e-5, 1e2)
    ax[1].set_ylim(1, 1e3)

    return ax


def plot_distribution(x):
    fig, ax = plt.subplots()
    x_axes = np.logspace(0, int(np.log10(max(x))) + 1, num=10)
    y, x_axes = np.histogram(x, bins=x_axes, density=True)
    x_axes = np.sqrt(x_axes[:-1] * x_axes[1:])
    ax.scatter(x_axes[y != 0], y[y != 0])
    perform_fit_over = y != 0
    popt, pcov = curve_fit(
        lineal,
        np.log10(x_axes[perform_fit_over]),
        np.log10(y[perform_fit_over]),
        maxfev=10000,
    )

    ax.plot(
        x_axes,
        10 ** lineal(np.log10(x_axes), a=popt[0], s=popt[1]),
        "r-",
        color="blue",
    )
    ax.text(
        0.3,
        0.7,
        r"$S(\tau) \approx \tau ^ {{ {:.2f} }} $".format(popt[0]),
        transform=ax.transAxes,
    )

    ax.text(
        0.1,
        0.5,
        r"$min {:.4g}  max {:.4g} $".format(min(x), max(x)),
        transform=ax.transAxes,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("mean {:.2g}".format(x.mean()))
