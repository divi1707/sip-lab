"""UI-independent orchestration: prepared data -> all inversions -> tidy results."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diagnostics import PrepSettings, PrepReport, prepare_spectrum
from inversion import (ColeColeSettings, DebyeSettings, Spectrum, bic, debye_decomposition, fit_cole_cole,
                       fit_power_law)


@dataclass
class RunConfig:
    prep: PrepSettings
    cc: ColeColeSettings
    debye: DebyeSettings
    run_cc: bool = True
    run_power: bool = True
    run_debye: bool = True


def run_all(df, cfg: RunConfig) -> dict:
    """Returns dict(spec, data, prep_report, cc, power, debye, errors).  Numerical failures of one method
    are captured in ``errors`` and do not abort the others."""
    spec, d, rep = prepare_spectrum(df, cfg.prep)
    out = dict(spec=spec, data=d, prep=rep, cc=None, power=None, debye=None, errors={})
    for key, flag, fn in (("power", cfg.run_power, lambda: fit_power_law(spec)),
                          ("cc", cfg.run_cc, lambda: fit_cole_cole(spec, cfg.cc)),
                          ("debye", cfg.run_debye, lambda: debye_decomposition(spec, cfg.debye))):
        if not flag:
            continue
        try:
            out[key] = fn()
        except Exception as e:  # noqa: BLE001 - reported to the user, never hidden
            out["errors"][key] = f"{type(e).__name__}: {e}"
    return out


def parameter_rows(out: dict) -> list[dict]:
    rows = []
    cc, pl, db = out.get("cc"), out.get("power"), out.get("debye")
    if cc:
        p, e = cc.params, cc.stderr
        rows.append(dict(model="Cole-Cole", parameter="rho0", value=p["rho0"], stderr=p["rho0"] * e["rho0_rel"][0], unit="Ohm m"))
        rows.append(dict(model="Cole-Cole", parameter="rho_inf", value=p["rho_inf"], stderr="", unit="Ohm m"))
        for k in range(len(p["m"])):
            rows += [dict(model="Cole-Cole", parameter=f"m{k+1}", value=p["m"][k], stderr=e["m"][k], unit="-"),
                     dict(model="Cole-Cole", parameter=f"tau{k+1}", value=p["tau"][k], stderr=f"x{10**e['log10tau'][k]:.3g} (multiplicative)", unit="s"),
                     dict(model="Cole-Cole", parameter=f"c{k+1}", value=p["c"][k], stderr=e["c"][k], unit="-")]
    if pl:
        for k, u in (("rho0", "Ohm m"), ("B", "-"), ("c", "-"), ("phase_1Hz_deg", "deg")):
            se = pl.stderr.get(k, pl.stderr.get("rho0_rel") * pl.params["rho0"] if k == "rho0" else "")
            rows.append(dict(model="Power-law", parameter=k, value=pl.params[k], stderr=se, unit=u))
    if db:
        rows += [dict(model="Debye", parameter="rho0", value=db.rho0, stderr="", unit="Ohm m"),
                 dict(model="Debye", parameter="m_total", value=db.derived["m_total"], stderr="", unit="-"),
                 dict(model="Debye", parameter="tau_mean", value=db.derived["tau_mean_s"], stderr="", unit="s"),
                 dict(model="Debye", parameter="lambda", value=db.lam, stderr="", unit="-")]
    return rows


def models_dict(out: dict) -> dict[str, np.ndarray]:
    m = {}
    if out.get("power"):
        m["Power-law"] = out["power"].model
    if out.get("cc"):
        m["Cole-Cole"] = out["cc"].model
    if out.get("debye"):
        m["Debye"] = out["debye"].model
    return m


def model_comparison(out: dict) -> list[dict]:
    rows = []
    for name, key in (("Power-law", "power"), ("Cole-Cole", "cc")):
        r = out.get(key)
        if r:
            s = r.stats
            rows.append({"model": name, "P": s["n_params"], "chi2_red": round(s["chi2_red"], 3),
                         "rms_amp_%": round(s["rms_amp_pct"], 3), "rms_phase_mrad": round(s["rms_phase_mrad"], 3),
                         "BIC": round(bic(s), 1)})
    if out.get("debye"):
        s = out["debye"].stats
        rows.append({"model": "Debye (regularised)", "P": "-", "chi2_red": round(s["chi2_red"], 3),
                     "rms_amp_%": round(s["rms_amp_pct"], 3), "rms_phase_mrad": round(s["rms_phase_mrad"], 3), "BIC": "-"})
    return rows
