"""SIP lab inversion – Streamlit app.   Run:  streamlit run app.py"""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import pandas as pd
import streamlit as st

from diagnostics import PrepSettings, prepare_spectrum, suggest_fmax
from export_results import figures_zip, parameters_csv, predicted_csv, report_txt
from inversion import ColeColeSettings, DebyeSettings
from pipeline import RunConfig, model_comparison, models_dict, parameter_rows, run_all
from res_parser import ParseError, parse_res
from synthetic import make_synthetic, write_synthetic_res
import visualization as viz

st.set_page_config(page_title="SIP lab inversion", layout="wide")
ss = st.session_state
ss.setdefault("cfg", {})
cfg: dict = ss["cfg"]
PAGES = ["Dashboard", "Data Preview", "Inversion Settings", "Run Inversion", "Results", "Export"]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.caption("1-D spectral inversion of a single laboratory SIP spectrum (Fuchs .res). "
                   "Not a spatial/2-D inversion.")


def keep(key, widget, *a, **k):
    """Render a widget whose value survives page changes (Streamlit drops state of hidden widgets)."""
    if "value" in k:
        k["value"] = cfg.get(key, k["value"])
    val = widget(*a, **k)
    cfg[key] = val
    return val


def load_bytes(raw: bytes, name: str, synthetic=False, truth=None):
    try:
        ss.res = parse_res(raw)
        ss.fname, ss.synthetic, ss.truth, ss.out = name, synthetic, truth, None
        ss.parse_error = None
    except ParseError as e:
        ss.res, ss.parse_error = None, str(e)


# ------------------------------------------------------------------ Dashboard
if page == "Dashboard":
    st.title("SIP lab inversion")
    st.markdown("Upload a **SIP-Fuchs III(+) `.res`** file. The app parses it, shows what was found, lets you "
                "correct geometry/sign/frequency band, and fits **Cole–Cole**, **power-law** and **Debye "
                "decomposition** models to the complex-resistivity spectrum, with diagnostics and exports.")
    up = st.file_uploader("Upload .res file", type=["res", "dat", "txt"])
    if up is not None and ss.get("fname") != up.name:
        load_bytes(up.getvalue(), up.name)
    c1, c2 = st.columns(2)
    if c1.button("Load SYNTHETIC demo (known truth)"):
        syn = make_synthetic()
        p = os.path.join(tempfile.gettempdir(), "synthetic_demo.res")
        write_synthetic_res(p, syn)
        load_bytes(open(p, "rb").read(), "synthetic_demo.res", True, syn["truth"])
    if ss.get("parse_error"):
        st.error(ss.parse_error)
    r = ss.get("res")
    if r is not None:
        if ss.synthetic:
            st.warning("SYNTHETIC data loaded – results are a numerical test, not a measurement. "
                       f"True parameters: {ss.truth}")
        st.success(f"Parsed **{ss.fname}**")
        st.write({k: v for k, v in r.header.items()})
        for i in r.issues:
            {"info": st.info, "warning": st.warning, "error": st.error}[i.level](i.message)
        g = r.groups()
        if not g.empty:
            st.subheader("Usable measurements found")
            st.dataframe(g, use_container_width=True)
            if len(g) == 1:
                st.caption("The file holds a single (reading, channel) spectrum – typical for a lab core measurement.")

# ------------------------------------------------------------------ Data preview
elif page == "Data Preview":
    st.title("Data preview")
    r = ss.get("res")
    if r is None:
        st.info("Upload a file on the Dashboard first.")
        st.stop()
    g = r.groups()
    if g.empty:
        st.error("No finite data rows.")
        st.stop()
    opts = [f"reading {a}, channel {b}" for a, b in zip(g.reading, g.channel)]
    choice = keep("group", st.selectbox, "Measurement", opts, index=0)
    if choice not in opts:
        choice = opts[0]
    rd, ch = int(g.reading[opts.index(choice)]), int(g.channel[opts.index(choice)])
    cfg["rd"], cfg["ch"] = rd, ch
    df = r.spectrum(rd, ch)
    st.write(f"{len(df)} valid frequencies, {df.freq_hz.min():.3g} – {df.freq_hz.max():.3g} Hz; "
             f"file K-factor = {df.k_file.iloc[0]:.5g} m")
    st.caption("Columns: freq_hz [Hz], rhoa_ohm_m [Ω·m, as stored], phase_deg [°], dR_pct [%], dP_deg [°], "
               "current_mA [mA], k_file [m]. Electrode layout blocks (TPos/RPos) in the header are the instrument's "
               "field-layout template and are not used for a lab sample.")
    st.dataframe(df[["freq_hz", "rhoa_ohm_m", "phase_deg", "dR_pct", "dP_deg", "current_mA", "k_file"]], use_container_width=True)
    nan_rows = int(r.data.rhoa_ohm_m.isna().sum())
    st.warning(f"{nan_rows} of {len(r.data)} rows in the file are NaN (channels/readings not measured) and are ignored.")
    if (df.rhoa_ohm_m < 0).all():
        st.warning("All stored resistivities are negative – see sign handling in Inversion Settings.")
    if df.freq_hz.duplicated().any():
        st.warning("Duplicate frequencies present.")
    if not r.freq_table.empty:
        with st.expander("Frequency table from header"):
            st.dataframe(r.freq_table)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].semilogx(df.freq_hz, df.rhoa_ohm_m.abs(), "ko-", ms=3); ax[0].set_ylabel("|Rhoa| (Ω·m)")
    ax[1].semilogx(df.freq_hz, df.phase_deg, "ko-", ms=3); ax[1].set_ylabel("phase (°)")
    for a in ax:
        a.set_xlabel("f (Hz)"); a.grid(alpha=0.3, which="both")
    fig.suptitle("Raw stored values (unscaled)" + (" [SYNTHETIC]" if ss.synthetic else ""))
    st.pyplot(fig)

# ------------------------------------------------------------------ Settings
elif page == "Inversion Settings":
    st.title("Inversion settings")
    if ss.get("res") is None or "rd" not in cfg:
        st.info("Load a file and open Data Preview once to choose the measurement.")
        st.stop()
    st.subheader("Geometry and conventions")
    use_k = keep("use_k", st.checkbox, "Override K-factor with sample geometric factor L/A", value=False,
                 help="Fuchs computes ρ = K·Z with K = A/L [m]. If the K stored in the file is a placeholder, give the "
                      "geometric factor of your sample as L/A [1/m]; ρ is rescaled by (1/(L/A))/K_file.")
    if use_k:
        keep("k_inv", st.number_input, "L/A  (1/m)", value=27.98, min_value=0.001, format="%.4f")
    keep("sign_mode", st.selectbox, "Amplitude sign", ["auto", "keep", "flip"], index=0,
         help="auto: take |ρ| if most stored values are negative (reversed potential polarity).")
    keep("phase_sign", st.selectbox, "Phase sign multiplier", [1.0, -1.0], index=0,
         help="+1: instrument phase already in resistivity space (negative = polarisable).")
    st.subheader("Data selection and weighting")
    keep("dp_filter", st.checkbox, "Drop points with dP > |phase|", value=True)
    c1, c2 = st.columns(2)
    keep("fmin", c1.number_input, "f_min (Hz, 0 = none)", value=0.0, min_value=0.0, format="%.6f")
    keep("fmax", c2.number_input, "f_max (Hz, 0 = none)", value=0.0, min_value=0.0, format="%.3f")
    keep("floor_amp", c1.number_input, "Amplitude error floor (%)", value=0.5, min_value=0.01)
    keep("floor_ph", c2.number_input, "Phase error floor (mrad)", value=1.0, min_value=0.01)
    if st.button("Suggest f_max from model consistency"):
        try:
            d = ss.res.spectrum(cfg["rd"], cfg["ch"])
            _, dd, _ = prepare_spectrum(d, PrepSettings(
                k_user_inv_m=cfg.get("k_inv") if use_k else None, sign_mode=cfg["sign_mode"],
                phase_sign=cfg["phase_sign"], fmin=cfg["fmin"] or None,
                max_dP_over_phase=1.0 if cfg["dp_filter"] else None))
            fm = suggest_fmax(dd, cfg["floor_amp"], cfg["floor_ph"])
            st.info(f"Largest f_max with power-law χ²_red ≤ 2: {fm} Hz" if fm else "No window with χ²_red ≤ 2 found.")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    st.subheader("Models to run")
    keep("run_power", st.checkbox, "Power-law (3 parameters)", value=True)
    keep("run_cc", st.checkbox, "Cole–Cole (Pelton)", value=True)
    keep("run_debye", st.checkbox, "Debye decomposition", value=True)
    st.subheader("Cole–Cole bounds")
    keep("n_terms", st.selectbox, "Number of Cole–Cole terms", [1, 2, 3], index=0)
    a, b = st.columns(2)
    keep("m_lo", a.number_input, "m min", value=1e-4, format="%.5f"); keep("m_hi", b.number_input, "m max", value=0.999, format="%.3f")
    keep("tau_lo", a.number_input, "log10 τ min (s)", value=-8.0); keep("tau_hi", b.number_input, "log10 τ max (s)", value=5.0)
    keep("c_lo", a.number_input, "c min", value=0.05); keep("c_hi", b.number_input, "c max", value=1.0)
    keep("starts", st.slider, "Random starts", 5, 100, value=40)
    st.subheader("Debye regularisation")
    auto = keep("lam_auto", st.checkbox, "Automatic λ (discrepancy principle, χ²_red ≈ 1)", value=True)
    if not auto:
        keep("lam", st.number_input, "λ", value=0.1, min_value=1e-6, format="%.6f")

# ------------------------------------------------------------------ Run
elif page == "Run Inversion":
    st.title("Run inversion")
    if ss.get("res") is None or "rd" not in cfg:
        st.info("Load a file and select a measurement first (Data Preview).")
        st.stop()
    if st.button("Start inversion", type="primary"):
        try:
            df = ss.res.spectrum(cfg["rd"], cfg["ch"])
            prep = PrepSettings(
                k_user_inv_m=cfg.get("k_inv") if cfg.get("use_k") else None, sign_mode=cfg.get("sign_mode", "auto"),
                phase_sign=cfg.get("phase_sign", 1.0), fmin=cfg.get("fmin") or None, fmax=cfg.get("fmax") or None,
                max_dP_over_phase=1.0 if cfg.get("dp_filter", True) else None,
                floor_amp_pct=cfg.get("floor_amp", 0.5), floor_phase_mrad=cfg.get("floor_ph", 1.0))
            cc = ColeColeSettings(n_terms=cfg.get("n_terms", 1), m_bounds=(cfg.get("m_lo", 1e-4), cfg.get("m_hi", 0.999)),
                                  tau_bounds=(10 ** cfg.get("tau_lo", -8.0), 10 ** cfg.get("tau_hi", 5.0)),
                                  c_bounds=(cfg.get("c_lo", 0.05), cfg.get("c_hi", 1.0)), n_starts=cfg.get("starts", 40))
            deb = DebyeSettings(lam=None if cfg.get("lam_auto", True) else cfg.get("lam", 0.1))
            rc = RunConfig(prep, cc, deb, cfg.get("run_cc", True), cfg.get("run_power", True), cfg.get("run_debye", True))
            t0 = time.perf_counter()
            with st.spinner("Inverting…"):
                ss.out = run_all(df, rc)
            ss.rc, ss.runtime = rc, time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            ss.out = None
            st.error(f"Inversion could not be run: {e}")
    out = ss.get("out")
    if out:
        st.success(f"Finished in {ss.runtime:.1f} s – {out['prep'].n_out} of {out['prep'].n_in} points used.")
        for lv, t in out["prep"].messages:
            {"info": st.info, "warning": st.warning}.get(lv, st.info)(t)
        for k, e in out["errors"].items():
            st.error(f"{k}: {e}")
        for key, nm in (("power", "Power-law"), ("cc", "Cole–Cole"), ("debye", "Debye")):
            r = out.get(key)
            if r:
                st.write(f"**{nm}**: solver {'OK' if r.success else 'FAILED / unphysical'} – χ²_red = {r.stats['chi2_red']:.2f}")
                for w in r.warnings:
                    st.warning(f"{nm}: {w}")

# ------------------------------------------------------------------ Results
elif page == "Results":
    st.title("Results")
    out = ss.get("out")
    if not out:
        st.info("Run an inversion first.")
        st.stop()
    syn = ss.get("synthetic", False)
    if syn:
        st.warning(f"SYNTHETIC data. Known truth: {ss.truth}")
    spec, models = out["spec"], models_dict(out)
    st.subheader("Model comparison")
    st.dataframe(pd.DataFrame(model_comparison(out)), use_container_width=True)
    st.caption("χ²_red = Σ((d−m)/σ)²/(N−P) over ln|ρ| and phase; rms_amp = RMS relative amplitude misfit; "
               "BIC = χ² + P·ln N (lower is better). χ²_red ≫ 1 means misfit beyond the (floored) errors.")
    st.subheader("Parameters")
    st.dataframe(pd.DataFrame(parameter_rows(out)), use_container_width=True)
    st.pyplot(viz.plot_spectrum(spec, {k: v for k, v in models.items()}, syn))
    c1, c2 = st.columns(2)
    sel = c1.selectbox("Residuals for", list(models))
    c1.pyplot(viz.plot_residuals(spec, models[sel], sel, syn))
    c2.pyplot(viz.plot_argand(spec, models, syn))
    if out.get("debye"):
        d = out["debye"]
        c3, c4 = st.columns(2)
        c3.pyplot(viz.plot_rtd(d.tau, d.m_j, syn, (spec.freq.min(), spec.freq.max())))
        c4.pyplot(viz.plot_lcurve(d.lam_curve, d.lam))
    st.info("No spatial section is shown: a single sample spectrum carries no spatial information.")

# ------------------------------------------------------------------ Export
elif page == "Export":
    st.title("Export")
    out = ss.get("out")
    if not out:
        st.info("Run an inversion first.")
        st.stop()
    syn = ss.get("synthetic", False)
    spec, models = out["spec"], models_dict(out)
    st.download_button("Predicted data (CSV)", predicted_csv(spec, models, syn), "sip_predicted.csv", "text/csv")
    st.download_button("Model parameters (CSV)", parameters_csv(parameter_rows(out)), "sip_parameters.csv", "text/csv")
    figs = {"spectrum": viz.plot_spectrum(spec, models, syn), "argand": viz.plot_argand(spec, models, syn)}
    for n, m in models.items():
        figs[f"residuals_{n}"] = viz.plot_residuals(spec, m, n, syn)
    if out.get("debye"):
        figs["rtd"] = viz.plot_rtd(out["debye"].tau, out["debye"].m_j, syn)
    st.download_button("Figures (PNG, zip)", figures_zip(figs), "sip_figures.zip", "application/zip")
    rc = ss.rc
    meta = {"file": ss.get("fname"), "K_file_m": out["prep"].k_file, "K_used_m": out["prep"].k_used,
            "prep": rc.prep, "cole_cole": rc.cc, "debye": rc.debye, "n_points": out["prep"].n_out}
    res = {}
    for key, nm in (("power", "power-law"), ("cc", "cole-cole"), ("debye", "debye")):
        r = out.get(key)
        if r:
            res[nm] = {"stats": r.stats, "warnings": r.warnings, "success": r.success,
                       **({"params": r.params, "stderr": r.stderr} if key != "debye" else {"derived": r.derived, "lambda": r.lam})}
    st.download_button("Report (TXT)", report_txt(meta, out["prep"].messages, res, syn), "sip_report.txt", "text/plain")
