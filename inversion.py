"""Inversion of a single complex-resistivity spectrum.

Two documented approaches (both 1-D spectral, NOT spatial):

1. ``fit_cole_cole``  - weighted non-linear least squares, multi-start, box bounds.
2. ``debye_decomposition`` - regularised non-negative linear inversion for a relaxation-time
   distribution (Nordsiek & Weller 2008).

Data vector  d = [ ln|rho| , phi ]   (phi in rad)
Weights      1/sigma, sigma_lnA = max(dR/100, floor_amp), sigma_phi = max(dP*pi/180, floor_phase)
Objective    Phi(p) = sum_i ((d_i - g_i(p))/sigma_i)^2   [+ lambda^2 ||D m||^2 for Debye]
Misfit       chi2_red = Phi / (N - P)   (N data, P free parameters)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from forward_model import TWO_PI, cole_cole, debye_kernel


class InversionError(RuntimeError):
    """Raised when the inversion cannot be carried out (bad input, numerical failure)."""


@dataclass
class Spectrum:
    freq: np.ndarray            # Hz, ascending
    rho: np.ndarray             # complex, Ohm m
    sig_lnA: np.ndarray         # sigma of ln|rho| (relative amplitude error)
    sig_phi: np.ndarray         # sigma of phase [rad]

    def __post_init__(self):
        n = len(self.freq)
        if n < 5:
            raise InversionError(f"Only {n} frequencies available; at least 5 are required.")
        if not (np.all(np.isfinite(self.freq)) and np.all(np.isfinite(self.rho))):
            raise InversionError("Spectrum contains non-finite values.")
        if np.any(self.freq <= 0) or np.any(np.abs(self.rho) <= 0):
            raise InversionError("Frequencies and |rho| must be positive.")

    @property
    def lnA(self) -> np.ndarray:
        return np.log(np.abs(self.rho))

    @property
    def phi(self) -> np.ndarray:
        return np.angle(self.rho)


def build_spectrum(freq, rho_abs, phase_deg, dR_pct, dP_deg, *, floor_amp_pct=0.5,
                   floor_phase_mrad=1.0) -> Spectrum:
    """Assemble a weighted Spectrum.  Instrument errors are often unrealistically small, so
    error floors are applied (defaults: 0.5 % amplitude, 1 mrad phase)."""
    f = np.asarray(freq, float)
    o = np.argsort(f)
    sa = np.maximum(np.nan_to_num(np.asarray(dR_pct, float), nan=0.0) / 100.0, floor_amp_pct / 100.0)
    sp = np.maximum(np.deg2rad(np.nan_to_num(np.asarray(dP_deg, float), nan=0.0)), floor_phase_mrad * 1e-3)
    rho = np.asarray(rho_abs, float) * np.exp(1j * np.deg2rad(np.asarray(phase_deg, float)))
    return Spectrum(f[o], rho[o], sa[o], sp[o])


def residuals(spec: Spectrum, model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalised residuals (data-model)/sigma for ln-amplitude and phase."""
    ra = (spec.lnA - np.log(np.abs(model))) / spec.sig_lnA
    rp = (spec.phi - np.angle(model)) / spec.sig_phi
    return ra, rp


def misfit_stats(spec: Spectrum, model: np.ndarray, n_params: int) -> dict:
    """Definitions: chi2 = sum r^2 ; chi2_red = chi2/(N-P) with N=2*nfreq ;
    rms_amp_pct = RMS of (|d|-|m|)/|d| ; rms_phase_mrad = RMS of (phi_d-phi_m)."""
    ra, rp = residuals(spec, model)
    n = 2 * len(spec.freq)
    chi2 = float(np.sum(ra**2) + np.sum(rp**2))
    dof = max(n - n_params, 1)
    r_all = np.concatenate([ra, rp])
    lag1 = float(np.corrcoef(ra[:-1], ra[1:])[0, 1]) if len(ra) > 3 else float("nan")
    lag1p = float(np.corrcoef(rp[:-1], rp[1:])[0, 1]) if len(rp) > 3 else float("nan")
    return dict(
        chi2=chi2, chi2_red=chi2 / dof, n_data=n, n_params=n_params,
        rms_amp_pct=float(100 * np.sqrt(np.mean(((np.abs(spec.rho) - np.abs(model)) / np.abs(spec.rho)) ** 2))),
        rms_phase_mrad=float(1e3 * np.sqrt(np.mean((spec.phi - np.angle(model)) ** 2))),
        max_abs_norm_res=float(np.max(np.abs(r_all))),
        lag1_autocorr_amp=lag1, lag1_autocorr_phase=lag1p,
    )


# --------------------------------------------------------------------------- Cole-Cole
@dataclass
class ColeColeSettings:
    n_terms: int = 1
    rho0_bounds: tuple[float, float] | None = None   # None -> automatic from data
    m_bounds: tuple[float, float] = (1e-4, 0.999)
    tau_bounds: tuple[float, float] = (1e-8, 1e5)    # s
    c_bounds: tuple[float, float] = (0.05, 1.0)
    n_starts: int = 40
    max_nfev: int = 400
    ftol: float = 1e-10
    seed: int = 0


@dataclass
class ColeColeResult:
    success: bool
    params: dict
    stderr: dict
    model: np.ndarray
    stats: dict
    runtime_s: float
    n_starts_converged: int
    warnings: list[str] = field(default_factory=list)
    status: str = ""
    condition_number: float = float("nan")
    settings: ColeColeSettings | None = None


def _unpack(p: np.ndarray, nt: int):
    rho0 = np.exp(p[0])
    m, tau, c = p[1:1 + nt], np.exp(p[1 + nt:1 + 2 * nt]), p[1 + 2 * nt:1 + 3 * nt]
    return rho0, m, tau, c


def _resid_vec(p: np.ndarray, spec: Spectrum, nt: int) -> np.ndarray:
    rho0, m, tau, c = _unpack(p, nt)
    mod = cole_cole(spec.freq, rho0, m, tau, c)
    ra, rp = residuals(spec, mod)
    return np.concatenate([ra, rp])


def fit_cole_cole(spec: Spectrum, s: ColeColeSettings | None = None) -> ColeColeResult:
    """Weighted NLLS of a multi-term Pelton Cole-Cole model with multi-start."""
    s = s or ColeColeSettings()
    t0 = time.perf_counter()
    nt = s.n_terms
    if nt not in (1, 2, 3):
        raise InversionError("n_terms must be 1, 2 or 3.")
    amp = np.abs(spec.rho)
    r_lo, r_hi = s.rho0_bounds or (0.5 * amp.min(), 10.0 * amp.max())
    if not (0 < r_lo < r_hi):
        raise InversionError("Invalid rho0 bounds.")
    lb = np.r_[np.log(r_lo), [s.m_bounds[0]] * nt, [np.log(s.tau_bounds[0])] * nt, [s.c_bounds[0]] * nt]
    ub = np.r_[np.log(r_hi), [s.m_bounds[1]] * nt, [np.log(s.tau_bounds[1])] * nt, [s.c_bounds[1]] * nt]
    if np.any(lb >= ub):
        raise InversionError("Lower bounds must be below upper bounds.")

    rng = np.random.default_rng(s.seed)
    # sensible start values: rho0 ~ largest |rho| / max, m from amplitude dispersion
    rho_hi_guess = np.clip(amp.max(), r_lo * 1.0001, r_hi * 0.9999)
    dm = np.clip(1.0 - amp.min() / amp.max(), 0.01, 0.9)
    best, n_conv = None, 0
    for k in range(s.n_starts):
        p0 = np.r_[np.log(rho_hi_guess * rng.uniform(0.9, 1.5)),
                   rng.uniform(0.01, 1, nt) * dm,
                   rng.uniform(np.log(s.tau_bounds[0]), np.log(s.tau_bounds[1]), nt),
                   rng.uniform(s.c_bounds[0], s.c_bounds[1], nt)]
        p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)
        try:
            sol = least_squares(_resid_vec, p0, bounds=(lb, ub), args=(spec, nt), method="trf",
                                x_scale="jac", ftol=s.ftol, xtol=1e-12, max_nfev=s.max_nfev)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if sol.status > 0:
            n_conv += 1
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise InversionError("All starts failed numerically.")

    p = best.x
    rho0, m, tau, c = _unpack(p, nt)
    if np.sum(m) >= 1.0:
        warn_sum = ["Sum of chargeabilities >= 1: unphysical (rho_inf <= 0)."]
    else:
        warn_sum = []
    model = cole_cole(spec.freq, rho0, m, tau, c)
    stats = misfit_stats(spec, model, len(p))

    # linearised parameter uncertainty (1 sigma), scaled by sqrt(max(1, chi2_red))
    J = best.jac
    sv = np.linalg.svd(J, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    try:
        cov = np.linalg.pinv(J.T @ J) * max(1.0, stats["chi2_red"])
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(p), np.nan)
    stderr = {"rho0_rel": se[0], "m": se[1:1 + nt], "log10tau": se[1 + nt:1 + 2 * nt] / np.log(10),
              "c": se[1 + 2 * nt:]}

    # failure / poor-constraint diagnostics
    warns = list(warn_sum)
    tol = 1e-3
    at = lambda v, lo, hi: bool(v <= lo * (1 + tol) + tol or v >= hi * (1 - tol) - tol)  # noqa: E731
    for k in range(nt):
        if at(m[k], *s.m_bounds):
            warns.append(f"Term {k+1}: chargeability m at a bound.")
        if np.log(tau[k]) <= np.log(s.tau_bounds[0]) + 0.01 or np.log(tau[k]) >= np.log(s.tau_bounds[1]) - 0.01:
            warns.append(f"Term {k+1}: relaxation time at a bound (peak outside the measured band?).")
        if c[k] <= s.c_bounds[0] + 1e-3 or c[k] >= s.c_bounds[1] - 1e-3:
            warns.append(f"Term {k+1}: exponent c at a bound.")
        if not (spec.freq.min() / 10 <= 1 / (TWO_PI * tau[k]) <= spec.freq.max() * 10):
            warns.append(f"Term {k+1}: 1/(2*pi*tau) = {1/(TWO_PI*tau[k]):.3g} Hz lies >1 decade outside the data band; tau is extrapolated.")
        if stderr["log10tau"][k] > 1.0:
            warns.append(f"Term {k+1}: tau uncertain by > 1 decade (poorly constrained).")
        if stderr["m"][k] > 0.5 * max(m[k], 1e-6):
            warns.append(f"Term {k+1}: chargeability uncertainty > 50 % of its value.")
    if cond > 1e8:
        warns.append(f"Ill-conditioned Jacobian (cond = {cond:.1e}): parameters are strongly correlated.")
    if stats["chi2_red"] > 3:
        warns.append(f"chi2_red = {stats['chi2_red']:.1f} > 3: the model does not explain the data within errors.")
    if np.isfinite(stats["lag1_autocorr_phase"]) and stats["lag1_autocorr_phase"] > 0.6:
        warns.append("Phase residuals are strongly autocorrelated: systematic misfit (wrong model or coupling).")
    if np.isfinite(stats["lag1_autocorr_amp"]) and stats["lag1_autocorr_amp"] > 0.6:
        warns.append("Amplitude residuals are strongly autocorrelated: systematic misfit (wrong model or drift).")
    if n_conv < max(2, s.n_starts // 10):
        warns.append(f"Only {n_conv}/{s.n_starts} starts converged: solution may be a local minimum.")
    if rho0 > 3 * amp.max():
        warns.append("rho0 is >3x the largest measured |rho|: DC level is an extrapolation.")

    ok = bool(best.status > 0 and not warn_sum)
    params = dict(rho0=float(rho0), rho_inf=float(rho0 * (1 - np.sum(m))), m_total=float(np.sum(m)),
                  m=m.tolist(), tau=tau.tolist(), c=c.tolist())
    return ColeColeResult(ok, params, {k: np.atleast_1d(v).tolist() for k, v in stderr.items()}, model,
                          stats, time.perf_counter() - t0, n_conv, warns, best.message, cond, s)


# --------------------------------------------------------------------------- Debye decomposition
@dataclass
class DebyeSettings:
    n_per_decade: int = 6
    tau_pad_decades: float = 1.0
    lam: float | None = None       # None -> discrepancy principle
    lam_grid: tuple[float, float, int] = (-4.0, 3.0, 36)   # log10 range
    target_chi2_red: float = 1.0


@dataclass
class DebyeResult:
    success: bool
    tau: np.ndarray
    m_j: np.ndarray         # chargeability per relaxation time (relative to rho0)
    rho0: float
    model: np.ndarray
    stats: dict
    lam: float
    derived: dict
    warnings: list[str]
    runtime_s: float
    lam_curve: np.ndarray    # columns: lambda, chi2_red, roughness


def _solve_debye(spec: Spectrum, tau: np.ndarray, lam: float):
    """Linear problem in x=[rho0, a_1..a_J], a_j = rho0*m_j >= 0 (complex data in Re/Im form)."""
    K = debye_kernel(spec.freq, tau)                      # F x J
    A = np.hstack([np.ones((len(spec.freq), 1)), -K])     # rho = rho0 - sum a_j K_j
    # Re/Im weights from amplitude/phase errors (first-order propagation; keeps the small phase
    # information from being swamped by the amplitude error)
    a, ph = np.abs(spec.rho), np.angle(spec.rho)
    sig_re = a * np.sqrt((spec.sig_lnA * np.cos(ph)) ** 2 + (spec.sig_phi * np.sin(ph)) ** 2)
    sig_im = a * np.sqrt((spec.sig_lnA * np.sin(ph)) ** 2 + (spec.sig_phi * np.cos(ph)) ** 2)
    wr, wi = 1.0 / sig_re, 1.0 / sig_im
    Ar = np.vstack([A.real * wr[:, None], A.imag * wi[:, None]])
    d = np.concatenate([spec.rho.real * wr, spec.rho.imag * wi])
    J = len(tau)
    # second-difference smoothing on a_j (scaled by mean |rho| so lambda is dimensionless)
    D = np.zeros((J - 2, J + 1))
    for i in range(J - 2):
        D[i, 1 + i:4 + i] = [1, -2, 1]
    scale = np.mean(np.abs(spec.rho))
    Aaug = np.vstack([Ar, lam * D / scale])
    daug = np.concatenate([d, np.zeros(J - 2)])
    lb = np.zeros(J + 1)
    lb[0] = 1e-12
    sol = lsq_linear(Aaug, daug, bounds=(lb, np.inf), method="bvls", lsmr_tol="auto", max_iter=2000)
    x = sol.x
    model = A @ x
    return x, model, float(np.linalg.norm(D @ x / scale))


def debye_decomposition(spec: Spectrum, s: DebyeSettings | None = None) -> DebyeResult:
    s = s or DebyeSettings()
    t0 = time.perf_counter()
    f = spec.freq
    lo = np.log10(1 / (TWO_PI * f.max())) - s.tau_pad_decades
    hi = np.log10(1 / (TWO_PI * f.min())) + s.tau_pad_decades
    nj = max(int(round((hi - lo) * s.n_per_decade)) + 1, 8)
    tau = np.logspace(lo, hi, nj)
    n_data = 2 * len(f)

    def chi2red(model):
        return misfit_stats(spec, model, 1)["chi2"] / max(n_data - 1, 1)

    grid = np.logspace(*s.lam_grid[:2], int(s.lam_grid[2])) if s.lam is None else np.array([s.lam])
    curve = []
    sols = []
    for lam in grid:
        x, model, rough = _solve_debye(spec, tau, lam)
        curve.append((lam, chi2red(model), rough))
        sols.append((x, model))
    curve = np.array(curve)
    if s.lam is None:
        ok_idx = np.where(curve[:, 1] <= s.target_chi2_red)[0]
        i = int(ok_idx[-1]) if len(ok_idx) else int(np.argmin(curve[:, 1]))   # largest lambda meeting target
    else:
        i = 0
    x, model = sols[i]
    rho0 = float(x[0])
    m_j = x[1:] / rho0
    stats = misfit_stats(spec, model, 1)
    # derived (Nordsiek & Weller 2008): total chargeability and mean (log-weighted) tau
    m_tot = float(np.sum(m_j))
    tau_mean = float(np.exp(np.sum(m_j * np.log(tau)) / m_tot)) if m_tot > 0 else float("nan")
    derived = dict(m_total=m_tot, tau_mean_s=tau_mean, rho_inf=float(rho0 * (1 - m_tot)))
    warns = []
    if s.lam is None and not len(np.where(curve[:, 1] <= s.target_chi2_red)[0]):
        warns.append(f"Target chi2_red={s.target_chi2_red} not reachable; using lambda with minimum misfit "
                     f"(chi2_red={curve[i,1]:.2f}).")
    if m_tot >= 1:
        warns.append("Total chargeability >= 1: unphysical.")
    if m_tot < 1e-3:
        warns.append("Total chargeability < 0.001: negligible polarisation; relaxation times are meaningless.")
    if stats["chi2_red"] > 3:
        warns.append(f"chi2_red = {stats['chi2_red']:.1f}: data not explained within errors.")
    edge = (m_j[:2].sum() + m_j[-2:].sum()) / max(m_tot, 1e-12)
    if m_tot > 0 and edge > 0.3:
        warns.append("More than 30 % of the polarisation sits at the edges of the tau grid: "
                     "the relaxation peak lies outside the measured band.")
    return DebyeResult(True, tau, m_j, rho0, model, stats, float(curve[i, 0]), derived, warns,
                       time.perf_counter() - t0, curve)


# --------------------------------------------------------------------------- Power-law (CPA) model
@dataclass
class PowerLawResult:
    success: bool
    params: dict
    stderr: dict
    model: np.ndarray
    stats: dict
    runtime_s: float
    warnings: list[str]
    condition_number: float


def power_law_model(freq: np.ndarray, rho0: float, B: float, c: float) -> np.ndarray:
    """rho*(w) = rho0 [1 - B (i w / w_ref)^c], w_ref = 2 pi * 1 Hz.

    This is the low-frequency limit (w*tau << 1) of a Pelton Cole-Cole term with B = m (w_ref tau)^c.
    It describes a broad, peak-less ("constant-phase-angle"-like) dispersion and has only 3
    well-determined parameters, whereas m, tau, c separately are degenerate when no phase peak
    is inside the measured band."""
    z = (1j * np.asarray(freq, float) / 1.0) ** c
    return rho0 * (1.0 - B * z)


def fit_power_law(spec: Spectrum, n_starts: int = 30, seed: int = 0) -> PowerLawResult:
    t0 = time.perf_counter()
    amp = np.abs(spec.rho)
    lb = np.array([np.log(0.5 * amp.min()), 1e-6, 0.01])
    ub = np.array([np.log(10 * amp.max()), 0.99, 1.0])

    def fun(p):
        mod = power_law_model(spec.freq, np.exp(p[0]), p[1], p[2])
        ra, rp = residuals(spec, mod)
        return np.concatenate([ra, rp])

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        p0 = np.array([np.log(amp.max() * rng.uniform(0.9, 1.5)), rng.uniform(0.001, 0.3), rng.uniform(0.05, 0.9)])
        p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)
        try:
            sol = least_squares(fun, p0, bounds=(lb, ub), x_scale="jac", ftol=1e-10, xtol=1e-12, max_nfev=400)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise InversionError("Power-law fit failed numerically.")
    rho0, B, c = float(np.exp(best.x[0])), float(best.x[1]), float(best.x[2])
    model = power_law_model(spec.freq, rho0, B, c)
    stats = misfit_stats(spec, model, 3)
    J = best.jac
    sv = np.linalg.svd(J, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J)) * max(1.0, stats["chi2_red"]), 0, None))
    warns = []
    if stats["chi2_red"] > 3:
        warns.append(f"chi2_red = {stats['chi2_red']:.1f} > 3: model does not explain the data within errors.")
    if c <= 0.011 or c >= 0.999:
        warns.append("Exponent c at a bound.")
    if cond > 1e8:
        warns.append(f"Ill-conditioned Jacobian (cond = {cond:.1e}).")
    ph1 = float(np.rad2deg(np.angle(power_law_model(np.array([1.0]), rho0, B, c))[0]))
    params = dict(rho0=rho0, B=B, c=c, phase_1Hz_deg=ph1)
    stderr = dict(rho0_rel=se[0], B=se[1], c=se[2])
    return PowerLawResult(bool(best.status > 0), params, stderr, model, stats, time.perf_counter() - t0, warns, cond)


def bic(stats: dict) -> float:
    """Bayesian information criterion for Gaussian errors: chi2 + P ln N (lower = better)."""
    return float(stats["chi2"] + stats["n_params"] * np.log(stats["n_data"]))
