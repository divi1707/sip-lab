# SIP lab inversion (Fuchs III `.res` → Cole–Cole / power-law / Debye)

Local Streamlit app for **laboratory spectral induced polarization** of one sample.
Pure NumPy/SciPy – no pyGIMLi, no compiled geophysics stack, so it installs the same on Apple Silicon and Intel Macs.

> **Scope (read this).** A `.res` file from a core measurement holds **one complex-resistivity spectrum**.
> The tool therefore does a **1-D spectral inversion** (parameters per sample). It does **not** do a 2-D/3-D spatial
> inversion and never draws a subsurface section.  A spatial SIP inversion needs electrode geometry and many
> quadrupoles; the `TPos/RPos/Reading` blocks in your file are only the instrument's default field-layout template.

## Install (macOS, fresh environment)
```bash
# Conda
conda create -n siplab python=3.11 -y && conda activate siplab
pip install -r requirements.txt
# or plain venv
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

cd sip_lab
streamlit run app.py
```
Run the tests: `python -m pytest -q`   (15 tests: parsing, invalid input, forward model, parameter recovery,
residuals, failure handling, exports, and an end-to-end app smoke test).

Troubleshooting: if `pip` tries to compile scipy, upgrade pip first (`pip install -U pip`) and make sure the
interpreter is arm64 (`python -c "import platform;print(platform.machine())"` → `arm64`). If the browser does not
open, go to http://localhost:8501. No Fortran/C++ toolchain is required.
Optional cross-check with other open-source tools (not required): BISIP (Bayesian SIP) or pyGIMLi's SIP utilities.

## Workflow
1. **Dashboard** – upload the `.res` file (or press *Load SYNTHETIC demo*). Parse status, header, usable measurements.
2. **Data Preview** – pick (reading, channel); table of valid rows, NaN count, raw plots.
3. **Inversion Settings** – K-factor override, amplitude/phase sign, frequency window, error floors, bounds, models.
4. **Run Inversion** – runs all selected models; shows warnings (bounds hit, ill-conditioning, poor fit).
5. **Results** – model comparison table, parameters (± linearised 1σ), spectrum, residuals, complex plane, Debye τ-distribution.
6. **Export** – CSV (predicted data, parameters), PNG zip, TXT report with settings, assumptions and solver status.

## File format (SIP-Fuchs III+ `.res`, verified on `results_B1.res`)
Latin-1 text, CRLF. Header → `[FrequencyParameterBegin..End]` (29 frequencies, 40 kHz–0.2 mHz) → layout template
(`[NumReading]`, `[TPosN]`, `[RPosN]`) → `[CalibrationFactorsBegin..End]` → one data block per reading with columns
`Reading Ch. … K.-Factor /m … Freq. /Hz Rhoa/Ohm*m Phase /° dR /% dP /° Current /mA Date Time`.
Unmeasured channels are `NaN`. In `results_B1.res` only reading 1 / channel 1 (26 frequencies) is finite (26 of 3 240 rows).

## Conventions and assumptions
* Model (Pelton et al. 1978, resistivity form): ρ*(ω) = ρ₀[1 − Σ m_k (1 − 1/(1+(iωτ_k)^c_k))], ω = 2πf.
  ρ₀ DC resistivity (Ω·m), m ∈ [0,1) chargeability, τ (s), c ∈ (0,1]. ρ_∞ = ρ₀(1−Σm).
* Phase φ = arg ρ*. A polarisable material has **negative** φ in resistivity space (what Fuchs stores). Flip with *Phase sign* if yours is the conductivity convention.
* **K-factor.** The instrument computes ρ = K·Z with K = A/L in **m** (column `K.-Factor /m`). If you enter a sample factor L/A in **1/m**, then K = 1/(L/A) and ρ is rescaled by K/K_file.
  For B1: L/A = 27.98 m⁻¹ → K = 0.03574 m vs 0.0350 m in the file (+2.1 %), consistent with the file holding a rounded/placeholder K. Enter your own value if it differs.
* **Negative Rhoa.** In `results_B1.res` every amplitude is negative while phase is a small negative angle. This is the signature of a reversed polarity of the potential leads; |ρ| is used and the phase is left unchanged (Sign option: auto/keep/flip). Check your wiring/instrument settings.
* **Data vector** d = [ln|ρ|, φ]; σ_lnA = max(dR/100, floor), σ_φ = max(dP, floor). Instrument errors are often unrealistically small, so floors (default 0.5 % and 1 mrad) are applied.
* **Misfit.** χ² = Σ((d−m)/σ)²; χ²_red = χ²/(N−P), N = 2·n_freq. RMS amplitude (%) and phase (mrad) are also reported, plus lag-1 autocorrelation of residuals (≫0 ⇒ systematic misfit). BIC = χ² + P ln N compares models.
* **Cole–Cole fit**: bounded trust-region NLLS (`scipy.optimize.least_squares`), 40 random starts, parameters in (ln ρ₀, m, ln τ, c); uncertainties from the Jacobian (linearised 1σ, scaled by √max(1,χ²_red)). Failure flags: parameter at bound, τ outside the measured band, >1-decade τ uncertainty, cond(J) > 1e8, χ²_red > 3, autocorrelated residuals, few converged starts.
* **Power-law model** ρ₀[1 − B (if/1 Hz)^c]: the ωτ≪1 limit of Cole–Cole with B = m(2π·1 Hz·τ)^c. Use it when no phase peak lies inside the band – m, τ, c are then degenerate but ρ₀, B, c are well determined.
* **Debye decomposition** (Nordsiek & Weller 2008): ρ = ρ₀ − Σ a_j(1−1/(1+iωτ_j)), a_j ≥ 0 (bounded least squares), second-difference smoothing, λ chosen by the discrepancy principle (largest λ with χ²_red ≤ 1, else minimum misfit). Outputs total chargeability and log-weighted mean τ.
* **Data selection.** Points with dP > |φ| are dropped (phase not distinguishable from 0); rows with identical ρ/φ/errors at different frequencies (stale instrument buffer) are removed; *Suggest f_max* returns the highest f_max for which a power-law fit stays consistent (χ²_red ≤ 2).

## Results obtained on `results_B1.res` (computed here, L/A = 27.98 m⁻¹)
After sign/K handling, removal of the two duplicated lowest-frequency rows and the four points with dP > |φ|, 20 points remain (0.09 Hz–40 kHz).
Using all 20 points no model fits within errors (power-law χ²_red 13, Cole–Cole 19): the spectrum above ≈1 kHz (phase −1.1° → −8.7° at 40 kHz) is not described by a single smooth dispersion and is probably affected by cable/EM coupling.
Restricting to f ≤ 750 Hz (suggested window, 14 points): power-law χ²_red = 1.97 (ρ₀ ≈ 60.5 Ω·m, c ≈ 0.19, φ(1 Hz) ≈ −0.29°), Cole–Cole χ²_red = 2.11 but **τ sits on its lower bound and m, τ are unconstrained** – there is no phase peak in the band, so Cole–Cole τ/m must not be interpreted. Reliable quantities for this sample: |ρ| ≈ 59–60 Ω·m near 1 Hz, phase ≈ −0.3° at 1 Hz, and a broad, peak-less dispersion with c ≈ 0.2.
These are results of this software on one file, not independent validation.

## Limitations
* Single spectrum only; no spatial inversion, no field geometry, no inductive/EM coupling correction.
* Cole–Cole parameters are only meaningful when a phase maximum lies inside the measured band.
* Error floors and the dP filter are choices that change χ²_red; report them with the results.
* Low-frequency points (< ≈0.1 Hz in B1) show drift (9 min per frequency) and are filtered by the dP rule.

## References
Pelton, Ward, Hallof, Sill, Nelson (1978) Geophysics 43, 588–609 · Nordsiek & Weller (2008) Geophys. J. Int., *A new approach to fitting induced-polarization spectra* · Radic Research SIP-Fuchs III (ρ = K·Z).
