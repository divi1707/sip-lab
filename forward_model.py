"""Forward models for laboratory SIP (single sample, 1-D spectral).

Convention (resistivity space, Pelton et al. 1978):

    rho*(w) = rho0 * [ 1 - sum_k m_k * (1 - 1 / (1 + (i w tau_k)^c_k)) ],  w = 2 pi f

* rho* = |rho| exp(i phi) with phi = arg(rho*).  A polarizable sample has phi < 0
  (voltage lags current), which is how the Fuchs instrument reports "Phase /deg".
* rho0 = DC resistivity [Ohm m]; rho_inf = rho0 (1 - sum m_k).
* 0 <= m < 1 chargeability, tau [s] relaxation time, 0 < c <= 1 exponent.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def cole_cole(freq: np.ndarray, rho0: float, m: np.ndarray | float,
              tau: np.ndarray | float, c: np.ndarray | float) -> np.ndarray:
    """Complex resistivity of a (multi-term) Pelton Cole-Cole model."""
    f = np.asarray(freq, dtype=float)
    m, tau, c = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (m, tau, c))
    if not (m.shape == tau.shape == c.shape):
        raise ValueError("m, tau and c must have the same length.")
    if np.any(tau <= 0) or np.any(c <= 0) or rho0 <= 0:
        raise ValueError("rho0, tau and c must be positive.")
    iwt = (1j * TWO_PI * f[:, None] * tau[None, :]) ** c[None, :]
    return rho0 * (1.0 - np.sum(m[None, :] * (1.0 - 1.0 / (1.0 + iwt)), axis=1))


def debye_kernel(freq: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Matrix K[f, j] = 1 - 1/(1 + i w tau_j) used in Debye decomposition."""
    iwt = 1j * TWO_PI * np.asarray(freq, float)[:, None] * np.asarray(tau, float)[None, :]
    return 1.0 - 1.0 / (1.0 + iwt)


def amp_phase(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return |z| and arg(z) in radians."""
    return np.abs(z), np.angle(z)
