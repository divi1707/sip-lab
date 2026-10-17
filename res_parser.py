"""Parser for SIP-Fuchs III(+) ``.res`` export files (Radic Research).

File structure (verified on a real ``results_B1.res``):

* free-text header (instrument, date, time, flags)
* ``[FrequencyParameterBegin] ... [FrequencyParameterEnd]`` table
* ``[NumReading]``/``[ReadingBegin]`` + ``[TPosN]``/``[RPosN]`` -> *field layout template*
  (for lab core measurements this is only the instrument default, NOT the sample geometry)
* ``[CalibrationFactorsBegin] ... End]`` -> channel calibration
* one data block per reading, each starting with a line that begins ``Reading   Ch.``
  and containing ``Rhoa``.  Rows: reading, channel, gains, flags, K-factor [m], stacks,
  frequency [Hz], Rhoa [Ohm m], phase [deg], dR [%], dP [deg], current [mA], date, time.
  Channels that were not measured are written as ``NaN``.

The parser never guesses silently: every anomaly is returned as a ``ParseIssue``.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATA_COLUMNS = [
    "reading", "channel", "tgain", "rgain", "ccal", "f50", "f60", "drift", "uzi",
    "k_file", "stacks", "freq_hz", "rhoa_ohm_m", "phase_deg", "dR_pct", "dP_deg",
    "current_mA", "datetime_raw",
]
NUMERIC = DATA_COLUMNS[:17]


class ParseError(ValueError):
    """Raised when the file cannot be interpreted as a Fuchs .res file."""


@dataclass
class ParseIssue:
    level: str  # "info" | "warning" | "error"
    message: str


@dataclass
class ResFile:
    header: dict[str, str]
    freq_table: pd.DataFrame
    data: pd.DataFrame            # all rows incl. NaN rows
    layout: dict[str, object]
    issues: list[ParseIssue] = field(default_factory=list)

    @property
    def valid(self) -> pd.DataFrame:
        """Rows with finite frequency, amplitude and phase."""
        d = self.data
        return d[np.isfinite(d.freq_hz) & np.isfinite(d.rhoa_ohm_m) & np.isfinite(d.phase_deg)]

    def groups(self) -> pd.DataFrame:
        """Table of (reading, channel) groups with number of valid points."""
        v = self.valid
        if v.empty:
            return pd.DataFrame(columns=["reading", "channel", "n_valid", "k_file"])
        g = v.groupby(["reading", "channel"]).agg(
            n_valid=("freq_hz", "size"), k_file=("k_file", "first"),
            fmin=("freq_hz", "min"), fmax=("freq_hz", "max")).reset_index()
        return g.sort_values("n_valid", ascending=False).reset_index(drop=True)

    def spectrum(self, reading: int, channel: int) -> pd.DataFrame:
        v = self.valid
        s = v[(v.reading == reading) & (v.channel == channel)]
        return s.sort_values("freq_hz", ascending=False).reset_index(drop=True)


def _decode(raw: bytes | str) -> str:
    if isinstance(raw, str):
        return raw
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ParseError("Cannot decode file (tried utf-8, latin-1).")


def _num(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        return float("nan")


def parse_res(raw: bytes | str) -> ResFile:
    """Parse the content of a Fuchs ``.res`` file."""
    text = _decode(raw).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    issues: list[ParseIssue] = []

    # ---- data blocks ------------------------------------------------------
    hdr_idx = [i for i, l in enumerate(lines) if l.lstrip().startswith("Reading") and "Rhoa" in l]
    if not hdr_idx:
        raise ParseError("No data block found (expected a line starting 'Reading ... Rhoa ...'). "
                         "This does not look like a SIP-Fuchs .res file.")
    rows: list[list] = []
    n_bad = 0
    for h in hdr_idx:
        for l in lines[h + 1:]:
            if not l.strip():
                break
            t = l.split(None, 17)
            if len(t) < 17:
                n_bad += 1
                continue
            rows.append([_num(x) for x in t[:17]] + [t[17].strip() if len(t) > 17 else ""])
    if n_bad:
        issues.append(ParseIssue("warning", f"{n_bad} malformed data rows were skipped."))
    data = pd.DataFrame(rows, columns=DATA_COLUMNS)
    for c in ("reading", "channel"):
        data[c] = data[c].astype("Int64")
    issues.append(ParseIssue("info", f"{len(hdr_idx)} data blocks, {len(data)} rows."))

    # ---- header -----------------------------------------------------------
    header: dict[str, str] = {}
    for l in lines[:20]:
        m = re.match(r"^(Instrument|Software Rev\.|Date|Time):\s*(.*)$", l)
        if m:
            header[m.group(1)] = m.group(2).strip()
        m = re.match(r"^\[(Simulationmode|Shunt|MaxPAVoltage|MaxPACurrent /mA|MultiFrequency|DFT)\]\s*(\S+)", l)
        if m:
            header[m.group(1)] = m.group(2)
    if "Spectral Induced Polarization" in text[:600]:
        header["Method"] = "Spectral Induced Polarization"

    # ---- frequency table --------------------------------------------------
    try:
        a = next(i for i, l in enumerate(lines) if "[FrequencyParameterBegin]" in l)
        b = next(i for i, l in enumerate(lines) if "[FrequencyParameterEnd]" in l)
        ft = [[_num(x) for x in l.split()[:7]] for l in lines[a + 1:b] if l.strip()]
        freq_table = pd.DataFrame(ft, columns=["freq_hz", "data_rate_hz", "samples_per_period",
                                               "partial_stacks", "stacks", "delay_s", "active"])
    except StopIteration:
        freq_table = pd.DataFrame()
        issues.append(ParseIssue("warning", "No frequency parameter table found."))

    # ---- layout template --------------------------------------------------
    layout: dict[str, object] = {}
    for key in ("NumReading", "NumTPos", "NumRPos"):
        m = re.search(rf"\[{key}\]\s*(\d+)", text)
        if m:
            layout[key] = int(m.group(1))

    res = ResFile(header, freq_table, data, layout, issues)
    n_valid = len(res.valid)
    if n_valid == 0:
        issues.append(ParseIssue("error", "All data rows are NaN: no usable measurement in the file."))
    else:
        issues.append(ParseIssue("info", f"{n_valid} of {len(data)} rows contain finite data "
                                         f"({100 * n_valid / len(data):.2f} %)."))
    return res


def parse_file(path: str) -> ResFile:
    with open(path, "rb") as fh:
        return parse_res(fh.read())
