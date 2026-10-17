"""Data-quality diagnostics and pre-processing of a Fuchs spectrum."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from inversion import Spectrum, build_spectrum


@dataclass
class PrepSettings:
    k_user_inv_m: float | None = None   # sample geometric factor L/A [1/m]; None -> keep file K
    sign_mode: str = "auto"             # "auto" | "keep" | "flip"  (amplitude sign)
    phase_sign: float = 1.0             # multiply phase by +1/-1 (convention switch)
    fmin: float | None = None
    fmax: float | None = None
    max_dP_over_phase: float | None = None   # drop points where dP > this * |phase| (None = off)
    floor_amp_pct: float = 0.5
    floor_phase_mrad: float = 1.0


@dataclass
class PrepReport:
    messages: list[tuple[str, str]] = field(default_factory=list)  # (level, text)
    k_file: float = float("nan")
    k_used: float = float("nan")
    k_scale: float = 1.0
    sign_flipped: bool = False
    n_in: int = 0
    n_out: int = 0

    def add(self, level: str, text: str):
        self.messages.append((level, text))


def prepare_spectrum(df: pd.DataFrame, s: PrepSettings) -> tuple[Spectrum, pd.DataFrame, PrepReport]:
    """Turn a parsed (reading, channel) table into a weighted ``Spectrum``.

    * Geometry: the instrument computes rho = K_file * Z.  If the user supplies L/A [1/m]
      the geometric factor is K = A/L = 1/(L/A) [m] and rho is rescaled by K/K_file.
    * Sign: a negative amplitude (reversed potential-electrode polarity) is mapped to |rho|;
      the phase is left unchanged because the instrument already reports it modulo 180 deg.
    """
    rep = PrepReport(n_in=len(df))
    d = df.copy().reset_index(drop=True)
    if d.empty:
        raise ValueError("No valid rows for this reading/channel.")
    k_file = float(d.k_file.iloc[0]) if np.isfinite(d.k_file.iloc[0]) else float("nan")
    rep.k_file = k_file
    if s.k_user_inv_m is not None:
        if s.k_user_inv_m <= 0:
            raise ValueError("Geometric factor L/A must be positive.")
        k_used = 1.0 / s.k_user_inv_m
        if not np.isfinite(k_file) or k_file <= 0:
            raise ValueError("File K-factor is invalid; cannot rescale.")
        rep.k_scale = k_used / k_file
        rep.k_used = k_used
        d["rhoa_ohm_m"] = d.rhoa_ohm_m * rep.k_scale
        rel = abs(rep.k_scale - 1)
        rep.add("warning" if rel > 0.01 else "info",
                f"K-factor: file K = {k_file:.5g} m, user L/A = {s.k_user_inv_m:.5g} 1/m -> K = {k_used:.5g} m; "
                f"rho rescaled by {rep.k_scale:.4f} ({100*rel:.1f} % change).")
    else:
        rep.k_used = k_file
        rep.add("info", f"Using the K-factor stored in the file (K = {k_file:.5g} m). Verify it matches your sample.")

    neg = d.rhoa_ohm_m < 0
    if s.sign_mode == "flip" or (s.sign_mode == "auto" and neg.mean() > 0.5):
        d["rhoa_ohm_m"] = -d.rhoa_ohm_m
        rep.sign_flipped = True
        rep.add("warning", "Amplitudes were negative (typical of reversed potential-electrode polarity): "
                           "|rho| is used. Phase left unchanged.")
    if (d.rhoa_ohm_m <= 0).any():
        raise ValueError("Non-positive resistivities remain after sign handling; choose the sign option manually.")

    d["phase_deg"] = d.phase_deg * s.phase_sign
    if d.phase_deg.median() > 0:
        rep.add("warning", "Median phase is positive. In resistivity space polarisable samples have NEGATIVE phase; "
                           "if your data follow the conductivity convention set phase sign = -1.")

    # identical amplitude/phase/error at different frequencies = stale instrument buffer, not data
    key = d[["rhoa_ohm_m", "phase_deg", "dR_pct", "dP_deg"]].round(9).apply(tuple, axis=1)
    dup = key.duplicated(keep=False)
    if dup.any():
        rep.add("warning", f"{int(dup.sum())} rows carry identical rho/phase/error values at different frequencies "
                           f"({', '.join(f'{x:.3g}' for x in d.freq_hz[dup])} Hz): looks like a repeated/stale reading; removed.")
        d = d[~dup]
    if s.fmin is not None:
        d = d[d.freq_hz >= s.fmin]
    if s.fmax is not None:
        d = d[d.freq_hz <= s.fmax]
    if s.max_dP_over_phase is not None:
        bad = d.dP_deg > s.max_dP_over_phase * d.phase_deg.abs()
        if bad.any():
            rep.add("info", f"{int(bad.sum())} points dropped: dP > {s.max_dP_over_phase} x |phase| "
                            f"(f = {', '.join(f'{x:.3g}' for x in d.freq_hz[bad])} Hz).")
        d = d[~bad]
    # quality warnings
    if len(d):
        lowq = d[(d.dP_deg > d.phase_deg.abs())]
        if len(lowq):
            rep.add("warning", f"{len(lowq)} points have dP larger than |phase| (lowest f: {lowq.freq_hz.min():.3g} Hz): "
                               "phase there is not statistically different from zero.")
        if d.phase_deg.abs().max() < 0.05:
            rep.add("warning", "All |phase| < 0.05 deg: sample shows virtually no polarisation.")
        if d.freq_hz.duplicated().any():
            rep.add("warning", "Duplicate frequencies present.")
        hf = d[d.freq_hz >= 1e4]
        if len(hf) and hf.phase_deg.abs().max() > 3 * max(d[d.freq_hz < 1e3].phase_deg.abs().median(), 1e-3):
            rep.add("warning", "Phase grows sharply above ~10 kHz: probably cable/electrode (EM) coupling, "
                               "not sample polarisation. Consider setting f_max.")
    rep.n_out = len(d)
    spec = build_spectrum(d.freq_hz.values, d.rhoa_ohm_m.values, d.phase_deg.values,
                          d.dR_pct.values, d.dP_deg.values,
                          floor_amp_pct=s.floor_amp_pct, floor_phase_mrad=s.floor_phase_mrad)
    return spec, d.sort_values("freq_hz").reset_index(drop=True), rep


def suggest_fmax(df_prepared: pd.DataFrame, floor_amp_pct: float = 0.5, floor_phase_mrad: float = 1.0,
                 threshold: float = 2.0, min_points: int = 8) -> float | None:
    """Largest f_max for which a 3-parameter power-law fit is consistent with the data
    (chi2_red <= threshold).  Above this the spectrum is no longer smooth/causal in the sense of the
    model (typically cable/EM coupling or electrode effects).  Returns None if no window qualifies."""
    from inversion import fit_power_law, build_spectrum  # local import avoids cycles
    d = df_prepared.sort_values("freq_hz")
    best = None
    for fmax in d.freq_hz.values:
        sub = d[d.freq_hz <= fmax]
        if len(sub) < min_points:
            continue
        spec = build_spectrum(sub.freq_hz.values, sub.rhoa_ohm_m.values, sub.phase_deg.values, sub.dR_pct.values,
                              sub.dP_deg.values, floor_amp_pct=floor_amp_pct, floor_phase_mrad=floor_phase_mrad)
        if fit_power_law(spec, n_starts=8).stats["chi2_red"] <= threshold:
            best = float(fmax)
    return best
