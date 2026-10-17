"""Synthetic SIP data (clearly labelled SYNTHETIC) and a writer that mimics the Fuchs .res format."""
from __future__ import annotations

import numpy as np

from forward_model import cole_cole

FUCHS_FREQS = np.array([
    40000, 20000, 10000, 5000, 3000, 1500, 750, 375, 187.5, 93.75, 46.875, 23.4375, 11.71875, 5.859375,
    2.9296875, 1.46484375, 0.732422, 0.366211, 0.183105, 0.091553, 0.045776, 0.022888, 0.011444, 0.005722,
    0.002682, 0.001341])


def make_synthetic(rho0=80.0, m=0.25, tau=0.05, c=0.6, freqs=FUCHS_FREQS, amp_noise_pct=0.3,
                   phase_noise_mrad=0.3, seed=1):
    """Noisy Cole-Cole spectrum with KNOWN parameters.  Returns dict with arrays (freq descending)."""
    rng = np.random.default_rng(seed)
    f = np.asarray(freqs, float)
    z = cole_cole(f, rho0, [m], [tau], [c])
    amp = np.abs(z) * (1 + rng.normal(0, amp_noise_pct / 100, f.size))
    ph = np.angle(z) + rng.normal(0, phase_noise_mrad * 1e-3, f.size)
    return dict(freq=f, amp=amp, phase_deg=np.rad2deg(ph), dR_pct=np.full(f.size, amp_noise_pct),
                dP_deg=np.full(f.size, np.rad2deg(phase_noise_mrad * 1e-3)),
                truth=dict(rho0=rho0, m=m, tau=tau, c=c))


def write_synthetic_res(path: str, syn: dict, k_file: float = 0.035, negative_amplitude: bool = False) -> None:
    """Write a Fuchs-style .res file (single reading/channel, rest NaN) from a synthetic dict."""
    sgn = -1.0 if negative_amplitude else 1.0
    L = ["Instrument:    SYNTHETIC (generated for testing - NOT field/lab data)", "Software Rev.: 000000",
         "Spectral Induced Polarization", "Date:          1/1/2026", "Time:          0:00 AM", "",
         "[Simulationmode]      1", "[Shunt]               301.000000", "[MaxPAVoltage]      20.000000",
         "[MaxPACurrent /mA]  20.000000", "[MultiFrequency]      1", "[DFT]                 1", "",
         "   Frequency /Hz  Data Rate /Hz   S/P  Part. Stacks Delay /s Active", "[FrequencyParameterBegin]"]
    for f in syn["freq"]:
        L.append(f"  {f:.8f} 96000.00000000   384    100      1      0.0      1")
    L += ["[FrequencyParameterEnd]", "", "[LayoutType] ", "", "[NumReading]   2",
          "Reading  RU0  RU1", "[ReadingBegin]", "      1    1    1", "      2    2    1", "[ReadingEnd]", "",
          "[NumTPos] 1", "[TPos1]  4.0 8.0 0.0 -1.0 8.0 0.0 -40.0 8.0 -40.0 0.0 8.0 0.0 40.0 0.0 0.0 1.0", "",
          "[NumRPos] 1", "[RPos1]  1.0 8.0 0.0 -0.5 0.0 0.0 0.5", "", "", ""]
    hdr = ("Reading   Ch.   TGain   RGain  C.Cal. 50Hz F. 60Hz F.  Dri.F.   U=Z*I   K.-Factor /m   Part.       "
           "Freq. /Hz   Rhoa/Ohm*m   Phase /°     dR /%     dP /°  Current /mA  Date /ddmmyy  Time /hhmmss")
    for rd in (1, 2):
        L.append(hdr)
        for ch in (1, 2):
            for i, f in enumerate(syn["freq"]):
                if rd == 1 and ch == 1:
                    L.append(f"      1     1       1       1       1       0       0       0       1"
                             f"         {k_file:.4f}     100    {f:.6f} {sgn*syn['amp'][i]: .6E}"
                             f"    {syn['phase_deg'][i]:.4f}    {syn['dR_pct'][i]:.4f}    {syn['dP_deg'][i]:.4f}"
                             f"  1.2E-2    1. 1.2026      0:00: 0")
                else:
                    L.append(f"      {rd}     {ch}       1       1       1       0       0       0       1"
                             f"            NaN     NaN             NaN          NaN        NaN       NaN"
                             f"       NaN           NaN           NaN")
        L.append("")
    with open(path, "w", encoding="latin-1", newline="") as fh:
        fh.write("\r\n".join(L) + "\r\n")


if __name__ == "__main__":      # python synthetic.py  ->  sample_data/synthetic_demo.res
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "synthetic_demo.res")
    syn = make_synthetic()
    write_synthetic_res(out, syn, k_file=0.035, negative_amplitude=True)
    print("wrote", out, "truth:", syn["truth"])
