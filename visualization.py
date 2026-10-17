"""Matplotlib figures (no Streamlit dependency)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inversion import Spectrum, residuals


def _label(synthetic: bool) -> str:
    return " [SYNTHETIC DATA]" if synthetic else ""


def plot_spectrum(spec: Spectrum, models: dict[str, np.ndarray], synthetic=False):
    """Amplitude and phase (resistivity space) with error bars and model curves."""
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].errorbar(spec.freq, np.abs(spec.rho), yerr=np.abs(spec.rho) * spec.sig_lnA, fmt="ko", ms=4,
                   label="measured (error incl. floor)")
    ax[1].errorbar(spec.freq, 1e3 * spec.phi, yerr=1e3 * spec.sig_phi, fmt="ko", ms=4)
    for (name, m), col in zip(models.items(), ["C3", "C0", "C2"]):
        ax[0].plot(spec.freq, np.abs(m), "-", color=col, label=f"predicted: {name}")
        ax[1].plot(spec.freq, 1e3 * np.angle(m), "-", color=col)
    ax[0].set_ylabel("|ρ*| (Ω·m)")
    ax[1].set_ylabel("phase φ (mrad)")
    ax[1].set_xlabel("frequency (Hz)")
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Complex resistivity spectrum" + _label(synthetic))
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_residuals(spec: Spectrum, model: np.ndarray, name: str, synthetic=False):
    ra, rp = residuals(spec, model)
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax[0].semilogx(spec.freq, ra, "o-", ms=4)
    ax[1].semilogx(spec.freq, rp, "s-", ms=4, color="C1")
    for a, t in zip(ax, ["amplitude: (ln|ρ|_obs − ln|ρ|_mod)/σ", "phase: (φ_obs − φ_mod)/σ"]):
        a.axhline(0, color="k", lw=0.8)
        a.axhspan(-1, 1, color="g", alpha=0.12)
        a.axhspan(-2, 2, color="g", alpha=0.07)
        a.set_ylabel(t, fontsize=8)
        a.grid(alpha=0.3, which="both")
    ax[1].set_xlabel("frequency (Hz)")
    ax[0].set_title(f"Normalised residuals – {name}" + _label(synthetic))
    fig.tight_layout()
    return fig


def plot_argand(spec: Spectrum, models: dict[str, np.ndarray], synthetic=False):
    """Complex-plane plot of conductivity-like Re(ρ) vs −Im(ρ)."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(spec.rho.real, -spec.rho.imag, "ko", ms=4, label="measured")
    for (n, m), col in zip(models.items(), ["C3", "C0", "C2"]):
        ax.plot(m.real, -m.imag, "-", color=col, label=n)
    ax.set_xlabel("Re ρ* (Ω·m)")
    ax.set_ylabel("−Im ρ* (Ω·m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Complex-plane plot" + _label(synthetic))
    fig.tight_layout()
    return fig


def plot_rtd(tau: np.ndarray, m_j: np.ndarray, synthetic=False, mark_band=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.semilogx(tau, m_j, "o-", ms=4)
    if mark_band is not None:
        lo, hi = mark_band  # relaxation times 1/(2 pi f) corresponding to the measured band
        ax.axvspan(1 / (2 * np.pi * hi), 1 / (2 * np.pi * lo), color="y", alpha=0.2, label="τ range of measured band")
        ax.legend(fontsize=8)
    ax.set_xlabel("relaxation time τ (s)")
    ax.set_ylabel("chargeability m_j")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Debye relaxation-time distribution" + _label(synthetic))
    fig.tight_layout()
    return fig


def plot_lcurve(curve: np.ndarray, lam_sel: float):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.loglog(curve[:, 0], curve[:, 1], "o-", ms=3)
    ax.axvline(lam_sel, color="r", ls="--", label=f"selected λ={lam_sel:.2g}")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("regularisation λ")
    ax.set_ylabel("χ²_red")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Misfit vs regularisation (Debye)")
    fig.tight_layout()
    return fig
