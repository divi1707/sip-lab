"""CSV / PNG / TXT export helpers."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

from inversion import Spectrum, residuals


def predicted_csv(spec: Spectrum, models: dict[str, np.ndarray], synthetic: bool = False) -> str:
    df = pd.DataFrame({"freq_Hz": spec.freq, "obs_abs_rho_ohm_m": np.abs(spec.rho),
                       "obs_phase_mrad": 1e3 * spec.phi, "sigma_lnA": spec.sig_lnA,
                       "sigma_phase_mrad": 1e3 * spec.sig_phi})
    for name, m in models.items():
        key = name.replace(" ", "_")
        ra, rp = residuals(spec, m)
        df[f"{key}_pred_abs_rho_ohm_m"] = np.abs(m)
        df[f"{key}_pred_phase_mrad"] = 1e3 * np.angle(m)
        df[f"{key}_res_amp_norm"] = ra
        df[f"{key}_res_phase_norm"] = rp
    df["data_origin"] = "SYNTHETIC" if synthetic else "measured"
    return df.to_csv(index=False)


def parameters_csv(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_csv(index=False)


def figures_zip(figs: dict[str, "matplotlib.figure.Figure"]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, fig in figs.items():
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=200)
            z.writestr(f"{name}.png", b.getvalue())
    return buf.getvalue()


def report_txt(meta: dict, prep_messages: list[tuple[str, str]], results: dict, synthetic: bool = False) -> str:
    """Plain-text report: settings, units, assumptions, solver status, parameters, warnings."""
    L = ["SIP LAB INVERSION REPORT", "=" * 60,
         f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
         "DATA ORIGIN: " + ("SYNTHETIC (known-truth test; not a field/lab result)" if synthetic else "measured file"), ""]
    L.append("INPUT / SETTINGS")
    for k, v in meta.items():
        L.append(f"  {k}: {v}")
    L += ["", "PRE-PROCESSING MESSAGES"] + [f"  [{lv}] {t}" for lv, t in prep_messages]
    L += ["", "CONVENTIONS & ASSUMPTIONS",
          "  rho*(w) = rho0[1 - sum m_k (1 - 1/(1+(i w tau_k)^c_k))], w = 2 pi f (Pelton et al. 1978)",
          "  phase = arg(rho*); polarisable material -> negative phase (resistivity space)",
          "  units: rho [Ohm m], f [Hz], tau [s], phase [mrad], m dimensionless",
          "  1-D spectral fit of ONE sample; no spatial/2-D inversion is performed",
          "  misfit: chi2 = sum((d-m)/sigma)^2 over ln|rho| and phase; chi2_red = chi2/(N-P)",
          "  parameter errors: linearised 1-sigma from Jacobian, scaled by sqrt(max(1,chi2_red))", ""]
    for name, r in results.items():
        L.append(f"RESULT: {name}")
        L.append(json.dumps(r, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
        L.append("")
    return "\n".join(L)
